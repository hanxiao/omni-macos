# OCR: the jina-ocr-v1 port, decoding, batching, the workspace engine

Moved out of CLAUDE.md on 2026-10-03, section text unchanged except where marked SUPERSEDED or
UPDATED. Dated findings, measurements and rejected options: read the section before touching
the code it names.

## OCR add-on (OmniKit/OCR/, docs/OCR.md)
- RETIRED 2026-10-02, once their results were recorded below: `--vision-prefetch`, `--probe-vision`,
  `--pipeline`, `--pipeline-host`, `--opener`, `--adaptive-draft`, `--probe-gemm`, `--probe-ane` and
  the OMNI_OCR_FUSED_MOE / FAST_RMSNORM / LM_SDPA arms. The notes below still name them; the code is
  in git history before that date. `--probe-qmm` stays (it reproduces a live MLX bug).
- Optional jina-ocr-v1 port. NOT downloaded unless the user asks, NOT on the index/search path.
- The numeric oracle is the ORIGINAL HF checkpoint at its shipped bfloat16, via torch/MPS - not
  the MLX python port and not an fp32 upcast. `Tools/ocr/ref_dump.py` produces it.
- Grade with `ocr-verify` on COMPLETE pages (natural EOS), and always against the HORIZON: the
  first character where torch's own bf16 and fp32 runs disagree. Beyond it there is no canonical
  text, so "exact" claims must state the compared range.
- Report prefix-exactness AND CER. They disagree: a build can "diverge at char 482 of 690" over
  one letter inside an HTML attribute (CER 0.0014).
- 4-bit is not a speed lever here. Decode is fixed-per-launch-latency bound at these skinny
  shapes, so the wins came from the fused gather-matmul MoE dispatch (+17%), 8-bit
  shared-expert/dense-MLP packs (+12%) and FastMTP speculation (+13-23%), not from narrowing
  the routed experts. 4-bit routed experts score CER 0.25 on handwriting - do not ship them.
- GRADE ON bench/hard2 (Tools/ocr/make_hard_pages.py), not on clean synthetic type. Easy pages
  agree whatever you do to the weights; the handwriting/receipt/spreadsheet/degraded set is what
  separated the builds and reversed a shipping decision.
- MLX 0.31.3 `quantizedMM(transpose: false)` is WRONG at exactly M=2 and M=3 (rel err ~1.5).
  `safeQuantizedMM` pads to 4. Reproduce with `ocr-verify --probe-qmm`. UPDATED 2026-10-03: every
  loaded pack is re-grouped to `transpose: true`, which has no such bug, so the padding only runs
  for a pack built outside the loader (see "MLX 0.32").
- Documents are N single-page requests. Multi-image prompts make the model transcribe only the
  LAST image (measured against torch by the predecessor). Never batch pages into one prompt.
- Do NOT cap output tokens with a constant. `OCRTokenBudget` derives it from the 32k context
  window and Metal's working set (31753/page here); a fixed 1024 silently truncated real pages.
- In-process page concurrency saturates at 2 lanes for +5%; PROCESSES scale (1.64x at 4, 1.59x on
  a 40-page doc) because MLX submits through one command queue per process. OCRWorkerPool does
  this, sized from Metal's working set. Pull pages from a QUEUE - round-robin lands periodic page
  types on one lane (measured 1.14x vs 1.41x).
- The draft chain runs WITHOUT returning to the CPU. `.item()` on each drafted token forced an
  evaluate-and-synchronise per draft; the id is only needed to look up an embedding, which can be
  done with it still on the GPU. k+2 blocking round trips per cycle became one. +7% and the
  document digest is unchanged (e6193a48ab3e331e): 201/271/215 -> 214/291/227 tok/s.
- `ocr-verify --pdf` IGNORED `--draft` until 2026-09-09, so any k sweep run through the document
  path before that measured the default every time and could only report a flat line.
- k curve RE-MEASURED after the sync removal (long_scan, aggregate): 174/185/188/180/181 at
  k=2..6. On bench/hard2 k=3 and k=4 tie at 169. Default stays 3 - one synthetic document is not
  grounds to move it, per the grading rule above.
- k=5 CHANGED THE OUTPUT on hard2 (digest 61dd84f8 vs d2d17e6a). RESOLVED 2026-10-03, and the
  guess here (the loop guard) was wrong: verify rows computed different bits from the greedy step
  for the same token, because MLX picks kernels by row count. Speculative output is now byte-
  identical to greedy at every k - see "Exact verify rows" under "MLX 0.32".
- Where the remaining time is (long page, 1309 tok): greedy 177.5 tok/s, speculative 213.9 - only
  1.20x, against 1.49x on a 352-token page. At ~2.9 tokens/cycle the draft chain is about half the
  cycle, and its cost is dominated by the MTP head's projection over the full 129k vocabulary,
  three or four times per cycle. That is the lever for anything past ~250 tok/s. FR-Spec measured
  only +1.8%, which is hard to reconcile with that share - worth checking whether the shortlist
  actually reduces the weight READ or merely slices a quantized matrix lazily. UPDATED: it was
  measuring nothing; fixed and ON, next bullet but one.
- Measured and rejected: `MLX_METAL_PREALLOCATE=1` (noise: 214.8/290.4/229.0 against
  214.4/290.6/227.0). Fused MoE dispatch is still right at n=5 (188 aggregate, against 132 never
  fused and 183 fused only at n<=16).
- FR-Spec draft-vocab shortlist is ON at 32768 (was "measured and rejected at +1.8%"). That
  measurement was of nothing: `draftHeadSlice` required `case .plain`, which no shipped build
  satisfies because the head is a PACK, and it sliced `lmHead` while `draftLogits` prefers
  `mtp.head`. Made to work on packs (groups run along the OUTPUT axis, so a vocabulary prefix is a
  contiguous slice of w/scales/biases, guarded on divisibility) it is worth +7.6% aggregate and
  +17% on a long page. 32768 is the peak: 199 aggregate against 185 full, 197 at 16384, 192 at 8192.
- A shortlist CANNOT change the output by construction - a drafted token still faces the target's
  full-vocabulary verification. What it can change is acceptance, and any acceptance change moves
  the batch shape of the verify forward, whose matmul reduction order differs with M. On long_scan
  the digest is identical at every shortlist size; on hard2 one token in ~1900 flips. That is the
  same floating-point tie-flip every speculative setting has relative to greedy, which is why the
  gate here is CER against the torch oracle and NOT digest equality. That CER run is still owed.
  SUPERSEDED 2026-10-03: verify rows are now exact, so acceptance can no longer change the output
  and the shortlist provably cannot either.
- Adaptive draft length was REJECTED because changing k mid-decode changed the output (hard2
  fafb7630 vs d2d17e6a). Same cause as the k=5 anomaly, now fixed, so that objection is gone; it
  has not been rebuilt or re-measured since.
- CURRENT BASELINE (2026-10-03, after "MLX 0.32"): 234 tok/s single at k = 3 (101.7 s, digest
  a35ef0f9c8fe8ad, the same at every k and greedy), 552 at width 32 (43.2 s, f1ed744e12ae453).
  Hard2: greedy 7/10 exact, mean CER 0.0087; continuous width 10: 8/10 (or 7/10), 0.0086-0.0087.
- 40-page long_scan BASELINE on this M3 Ultra, single process: 194 aggregate tok/s, 123.1 s,
  digest 772db0f94e0ae106, 62014 chars. Every throughput claim below is against that, and the
  digest is the quality gate - it stayed identical through every measurement in this round.
- PROCESS POOL, measured end to end on the 40-page doc: 194 / 284 / 321 / 323 / 322 at 1 / 2 / 4 /
  6 / 8 workers. It SATURATES at 4 - the GPU is the wall past that, not memory. Output identical at
  every count. But it is a big-machine win only: each worker holds its own 4.5 GB of weights, so
  `recommendedWorkers` returns 1 on a 16 GB laptop and the whole gain is zero for the machines that
  need it most. Do not quote it as the headline number.
- `--processes N --draft K` measured k = 3 at EVERY K until 2026-09-09: `OCRWorkerPool.init`
  defaults `draftLength: 3` and ocr-verify never passed it. The flat line (321/321/321) is the
  signature. Re-measured properly: 277 / 307 / 321 / 309 / 243 at k = 1 / 2 / 3 / 4 / 5, so k = 3
  is still the peak under saturation - speculation does NOT become a net loss when the GPU is busy,
  which was the hypothesis and it was wrong.
- TTFT BREAKDOWN, measured with `--report-prefill`: vision 336-413 ms, language prefill 313-317 ms,
  prompt embed ~0 ms, MTP priming 2-3 ms. 336 + 317 = 653 = the reported TTFT exactly. So prefill
  is half vision and half language, and the draft head's priming pass - the obvious suspect - costs
  nothing. Quantizing the vision tower would not help either: SAM-ViT-B + CLIP-L is ~800 MB read
  once per page, so it is compute-bound, not bandwidth-bound.
- `--vision-prefetch` re-measured on 40 pages: 197 against 194, and per-page TTFT 656 -> 595 ms. It
  reschedules work rather than removing it, which is why it is ~0 on a busy GPU. Confirmed, still off.
- BATCHED DECODE IS THE ANSWER, and it is the one that works on a laptop. B pages decode together
  through ONE copy of the weights (`OCRBatchKVCache`, `forwardBatch`, `transcribeBatched`, and
  `ocr-verify --pdf --batch B`). Measured on the 40-page doc, all with the SAME digest
  772db0f94e0ae106 as the single-process baseline - byte-identical output at every width:
  194 (single, speculative) / 158 (B=2) / 185 (B=4) / 260 (B=8) / 339 (B=16) / 485 (B=40).
  2.5x the baseline and past the 400 target, greedy, on one copy of the weights.
- The scaling is NOT linear at small B and that is the MoE: 8 tokens routing to 8 different sets of
  6 experts read 8x the expert weight, exactly as if they had run separately. Attention, the dense
  MLP, the shared expert and the LM head amortise from B=2; the routed experts only amortise once
  B is wide enough that the batch touches most of the 64 anyway. That crossover is why B=40 is
  worth 1.9x of B=8 and B=4 is worth almost nothing. Do not conclude from a narrow batch that
  batching does not work here - that was concluded mid-measurement and it was wrong.
- The batch width is SIZED, not fixed: `OCRBatchPlan.recommendedWidth` from Metal's working set,
  capped at 32 and floored at 8. Measured at the widths it actually picks: 401 on this M3 Ultra
  (32) and 295 on a 16 GB laptop's share (11), against 194 - and that same laptop affords exactly
  ONE worker process, i.e. no gain at all from the pool. Below 8 it returns 1 and the single
  speculative path runs instead, because a narrow batch is a regression (158 at B=2, 185 at B=4).
- Batch WIDTH is a memory decision, like the worker count: KV is 65 KB/token, a page runs to ~2300
  tokens, so a slot costs ~150 MB. B=16 is ~7 GB with the weights and comfortable on 16 GB; B=40 is
  ~10.5 GB and is not.
- THE BATCHING ARITHMETIC, from numbers already measured here: the language prefill carries 1007
  tokens in 317 ms (3177 tok/s) while a decode step carries 1 token in ~4.3 ms (234 tok/s). Same
  weights, same layers - a forward that carries many tokens costs ~13x less PER TOKEN. That is why
  batching pages inside ONE process is the real lever and worker processes are not: batching
  amortises both bounds (fixed per-launch latency, and reading the active experts once per step)
  over B sequences while holding one copy of the weights, so it works on a 16 GB machine.
- WIRED INTO THE APP: a run decodes in groups, sized by `OCRBatchPlan` or by "Pages at once" in
  the OCR settings tab (0 = Automatic, 1 = the page-at-a-time path, then 8/12/16/24/32). Measured
  IN THE APP on the same 40-page scan, final chip readings: 182 tok/s / 2:09 at one page at a time,
  245 / 1:39 at 8, 323 / 1:15 at 16, 381 / 1:03 at 32. That is 4-6% under the headless figures
  (193 / 259 / 339 / 399 at the same widths, all with digest 772db0f94e0ae106). NOT the stream's
  detokenizing: re-decoding every live page's whole id list at 24 Hz measured 457 tok/s against
  459 with no stream at all (2026-09-22), and an incremental detokenizer bought nothing.
- The chip's rate is the run's AGGREGATE - all tokens over the wall clock since the first page
  started decoding, model load excluded. It used to be the last streaming update's own rate, which
  is per-SLOT: at B = 32 that reads ~13 tok/s while the run is doing 400. Mid-run it also reads low
  for a different and honest reason - the first group's 32 vision towers and prefills are on the
  clock before much text exists (244 tok/s at 29 s, 381 at the end).
- A group is confined to ONE DROP BATCH, so a file opened mid-run does not join the group someone
  is watching. Four files dropped together are one batch and still decode as one group.
- A page SETTLES WHEN ITS OWN SEQUENCE STOPS, not when the group returns (`decodeGroup`'s
  `onFinish`). Page lengths in a group run from 75 to 1309 tokens, so holding every finished page
  until the slowest one stopped froze the whole workspace: on a ten-file drop every tab's progress
  ring sat at zero for the length of the run and then cleared at once. The group return is now a
  backstop for the pages that failed and for a callback that lost the race.
- PREFILL IS NOW THE DOMINANT COST, and it got that way by fixing decode. Per page it is fixed at
  ~653 ms while the decode share shrank with batch width: 21% of a run at width 1 (3.1 s/page),
  40% at width 16 (1.64 s/page). Measured at width 16 on 16 pages: 26.3 s total, of which
  16 x 335 ms of `preparePage` and 16 x ~317 ms of LM prefill. Any further work on throughput
  belongs here, not in the decode loop.
- `--report-prefill` now splits `preparePage` into HOST and TOWER. Measured per page: host 55 ms,
  tower 280 ms. The host half is `PILResample`, Pillow's fixed-point bicubic in pure Swift, and it
  must stay on the CPU in integer arithmetic - a float or Metal resize lands within a LSB or two
  and flips greedy ties, which is the failure mode the port exists to avoid. It is however the one
  part that is trivially parallel across cores and currently runs serially, one page at a time.
- BATCHING THE VISION TOWER ACROSS PAGES IS WORTH ~4%, measured, do not build it.
  `probeVisionScaling` (`ocr-verify --probe-vision`) times the real tower at 1/2/4/8/16/32 tiles:
  40.5 / 33.7 / 31.3 / 30.5 / 29.9 / 29.4 ms per tile, i.e. 1.37x from 1 to 32 and only 1.04x from
  8 to 32. A page already carries a 1024 global view (~101 ms) plus ~6 tiles at 640 (~30 ms each),
  so it is already past the knee. The vLLM `--mm-encoder-tp-mode data` trick that is worth 40% is
  about replicating the encoder across GPUS, not amortising it on one, and does not transfer.
- Corroborated upstream: ml-explore/mlx discussion #3829 measures per-frame VLM cost on an M3 Max
  and finds vision encode ~constant at ~75 ms/frame and "encoder-independent - optimizing the
  vision tower doesn't move the needle", with LM prefill dominating on capable models.
- GROUP-AHEAD PREFETCH: BUILT, MEASURED, NOT DEFAULT (`--pipeline`, `--pipeline-host`). Prefill
  the NEXT group on its own MLX stream while the current one decodes. It works exactly as
  designed - the stall it is meant to remove does fall, 14.6 -> 3.0 s at width 8 and 14.5 ->
  11.7 s at width 32 - and it buys almost nothing, because the GPU has no idle compute to absorb
  it: 259 -> 266 at width 8, 340 -> 350 at width 16, 400 -> 402 at width 32. Hidden work returns
  only 20-25% of its own time. THE HYPOTHESIS WAS WRONG: batch decode is not leaving the compute
  units idle for a compute-bound prefill to slot into, it is already saturating them. Host-only
  prefetch (rasterise and resample ahead, pure CPU) is worth 0.2-0.9 s of a whole run. Same
  conclusion `--vision-prefetch` reached on the single path, now confirmed for the batched one at
  17x the prefetch depth. Digest unchanged at every setting.
- MPSGRAPH / ANE FOR PREFILL: MEASURED AND REJECTED (`--probe-gemm`). BaseRT (arXiv 2607.00501)
  attributes uzu's prefill lead to MPSGraph's ability to reach the Neural Engine, so the raw op
  was timed at prefill shapes. MPSGraph fp16 runs at 0.84x to 1.04x of MLX - within noise, mostly
  slower: M1007xK1280xN1280 0.65 vs 0.64 ms, M4096xK1280xN1280 1.05 vs 0.88 ms. No ANE dispatch
  materialises. Note also that MLX fp32 matches or beats its own fp16 at these shapes (5-15
  TFLOPs against the chip's ceiling), so prefill's GEMMs are launch- and occupancy-bound rather
  than arithmetic-bound, and a faster GEMM kernel is not the lever. Our weights are quantized
  anyway, so an MPSGraph path would have to dequantize the model first.
- VISION CACHE: ADOPTED (`OCRVisionCache`). Visual features keyed by a 128-bit content hash of
  the pixel buffer. The tower and the resample depend only on the image - the prompt reaches the
  model as token ids and the pixels never appear in them - so `promptIDs(grid:prompt:)` is split
  out of `prepare` and a hit costs only the ids. Measured on the 40-page scan, same process:
  pass 1 59.5 s / 401 tok/s, pass 2 45.6 s / 523 tok/s, stall 14.6 -> 0.9 s, digest identical
  (772db0f94e0ae106). In the app, the same file dropped twice runs 80 pages at 473 tok/s against
  381 for a single pass. 4.4 MB a page; bounded at 256 MB and CHARGED TO `OCRBatchPlan`'s reserve
  so it costs a slot on paper rather than discovering one at runtime. Worth nothing on a first
  pass over distinct pages, which is why it is a cache: it pays for an edited prompt, a re-drop,
  and a page re-queued after a stop.
- `Prepared.global` is OPTIONAL because a cached page has features and no pixels. `visualFeatures`
  traps on nil and the stage dump throws - both are only ever driven from a freshly prepared page.
- THE WAIT BEFORE THE FIRST WORD IS PREFILL, NOT SHADER COMPILATION. Asked because a first run
  looked stuck for a long time: page 1's vision tower costs 364 ms against 284 ms for page 2, and
  a SECOND process measures the same 364, so Metal pipeline building is a ~80 ms one-off and
  there is nothing to precompile after download. The real cause is that every page of a group is
  prefilled before the group's first token exists - 10.0 s for a 32-page group here, roughly
  twice that on a laptop, with a label that did not move.
- THE RAMP is what fixes the opening wait, and continuous batching alone does NOT: it still
  admitted every row before the first step. Starting on `rampRows` (4) and admitting one more row
  per step moves those prefills off the front of the run without stranding the early pages in a
  narrow group, which is exactly what the static opener could not do.
- Do not slice the batch axis when every row is live. Writing `keys![0 ..< n, ...]` unconditionally
  in `appendStep` cost the in-place fast path and measured 282 aggregate tok/s against 401; the
  full-range form is kept for the common case and the slice only used while the batch is ramping.
- MEASURE BACK TO BACK. A long session leaves this machine measurably slower - the same committed
  build scored 401 tok/s early and 307 hours later - so a number from earlier in the day is not a
  baseline for a number now. Re-run the control.
- GROUPS ARE BALANCED, NOT GREEDY. 40 pages at width 32 used to be 32 + 8, and a narrow group
  costs nearly as much per step as a full one, so the stub was paid for twice. Splitting evenly
  (20 + 20) is free on throughput and better on latency: 401 tok/s against 399, first token at
  6.3 s against 10.0 s, digest unchanged.
- A NARROW OPENING GROUP IS MEASURED AND REJECTED. Four pages first puts words on screen in 1.2 s
  but costs 16% (399 -> 337, and 339 even with the rest balanced), because a group runs as long
  as its LONGEST page: an opener holding a 1309-token page decodes almost serially at width 4.
  No opener size escapes that. Continuous batching is the fix for this wait, not a smaller group.
- The workspace reports prefill progress ("Reading 12 of 20 pages") through `onPrefill`, because
  a wide group's prefill IS the wait and a still label reads as a hang.
- BATCH OCCUPANCY ON A REAL DOCUMENT IS 45%. Pages run 72 to 1310 tokens, so a static group runs
  for as many steps as its LONGEST page and spends most of them narrowed. That matters because a
  decode step is strongly SUB-LINEAR in its row count, measured at a 1024-token context with
  `--probe-decode-width`: 5.60 / 8.95 / 9.53 / 11.47 / 15.89 / 22.21 ms at 1 / 2 / 4 / 8 / 16 / 32
  rows. 32 rows cost 4x what one row does and carry 32x the tokens, so running narrow is expensive
  and the row, not the group, is the unit of work.
- CONTINUOUS BATCHING IS THE DEFAULT (`OCRRuntimeFlags.continuousBatch`, `--static` to turn it
  off in ocr-verify). A finished row is refilled with the next page instead of being dropped, and
  the batch RAMPS UP from 4 rows rather than prefilling every page first. Measured back to back
  on the 40-page scan: 282 -> 364 aggregate tok/s and first token 9.2 s -> 1.3 s. `OCRBatchKVCache`
  grew per-row `promptLen`/`startedAt` over a shared `cursor`, plus an `active` prefix so the
  batch can widen while decoding; a recycled row keeps its prompt at [0, promptLen) and its
  output from `startedAt`, and `mask()` hides the dead span between them.
- PROMPTS IN A GROUP ARE NOT THE SAME LENGTH, and assuming they are crashed the shipped app
  (EXC_BREAKPOINT, `OCRBatchKVCache.seed`, cursor 1007 against a 1197-token prompt) the first
  time several documents were dropped at once. A page's tile grid comes from its ASPECT RATIO,
  so a portrait scan and a squarer one produce different `imageIDs` counts and different prompt
  lengths, in the STATIC path as much as the continuous one. Two consequences: a row's output
  start cannot be fixed when it is seeded, because a later, longer prompt in the same group moves
  the shared cursor past it and silently turns that row's gap into valid history - it is assigned
  lazily, at the row's first `appendStep`; and a finished row can only be recycled when its new
  prompt fits under the cursor (`canAdmit`), otherwise the page waits for the next group.
  `OCRBatchCacheTests` pins all of it, including the 1007/1197 shape from the crash.
- Two traps found building it, both worth remembering. `rope(positions:)` was keyed
  "first-last-count", which is only unique while every row advances in lockstep - continuous
  batching makes two different position vectors collide on that key and silently swaps their
  rotations. Now keyed by the whole vector. And `transcribeBatched(pdfAt:)` pre-slices pages into
  groups of exactly `width`, so the scheduler was handed 32 pages with width 32, had nothing to
  admit, and degenerated to the static path while reporting a clean A/B of 403 vs 401 - a null
  result that was measuring nothing.
- THE GATE IT PASSED: `ocr-verify <model> <refRoot> --grade-batch [--batch N] [--static]` feeds the
  hard2 PNGs straight into `transcribeBatched` and scores CER against the same torch oracle the
  greedy gate uses, so the two are comparable page for page. Continuous at width 4 over 10 pages
  (6 rows recycled) scores 8/10 exact, mean CER 0.0086, against 7/10 and 0.0087 for the single
  path - every page identical except `receipt_faded`, which continuous gets EXACT where the
  single path had CER 0.0008. So the "199 mL" -> "199 m3" flip on long_scan is the near-tie class
  and, on the corpus that grades quality, the flips are neutral to positive.
- Two traps in building that gate, both of which made it measure nothing. `--grade-batch` must pass
  the SAME `--max-new` cap the greedy gate uses (1024): with the budget-derived cap the two hard2
  pages that never reach EOS run to 31753 tokens of repetition and score CER 29. And the runtime
  flags were assigned inside the `--pdf` branch, so `--continuous` was ignored by every other mode
  - the first three "continuous" gate runs were the static path under a continuous heading. Flags
  are parsed globally now.
- Converting hard2 to a PDF to reach the batch path is INVALID: at 200 dpi the 7pt columns turn to
  mush and the model loops (737 KB of repeated paragraphs for 30 pages). Use `--grade-batch`.
- IMAGE SIZE IS NOT A SPEED KNOB, measured: 12 pages at dpi 120 / 150 / 200 / 300 all run in
  22.5 s with an identical 4.5 s prefill stall. The reason is in `dynamicPreprocess`: it picks a
  tile grid from the ASPECT RATIO alone (`closestAspectRatio`, product capped at 9) and then
  resamples the WHOLE page to 640*tw x 640*th before cutting it up. The tower therefore sees the
  same pixel count for a given page shape whatever the source resolution - A4 portrait lands on
  2x3, so 6 tiles of 640 plus the 1024 global view, every time. Source resolution only buys detail
  and costs host resample time.
- 200 dpi is close to the floor, and not by accident: A4 at 200 dpi is 1654x2339 against a 1280x1920
  target, so the resample DOWNSAMPLES. At 150 it is 1240x1754 and starts upsampling the long edge.
  Digests agree at 150/200/300 and diverge at 120, which is where the detail loss first shows.
  Do not lower it for speed - there is none to win.
- Measured and rejected, do not re-derive: mlx-swift 0.31.4 (same qmm bug) and 0.32.x (fixes the
  qmm bug, costs ~4 ms a decode step - see "MLX 0.32"). DFlash/EAGLE trees need a draft model we cannot train here; ViT token merging breaks
  the fixed visual-token/prompt-slot contract.
- The OCR model does NOT live inside the app's MLX memory cap. It is loaded when the OCR toggle
  goes on and dropped when it goes off, and while it is loaded the compute cap is lifted. Charging
  4.53 GB of weights to a 6 GB budget whose buffer cache is a quarter of it costs more than half
  the throughput - measured in-app on the same pages: 92/136/101 tok/s capped against 202/274/218
  uncapped, where the same model outside the app does 201/271/215.
- `omniSetMemoryLimit(0)` did not reset `MLX.Memory.memoryLimit`, only the cache limit, so
  "Unlimited" never lifted a cap that had already been applied. Fixed; it is why the first attempt
  at the above changed nothing.
- Two primitives are measured-and-rejected: `MLXFast.rope` (faster and numerically WRONG - the
  checkpoint is Llama split-half, MLX is interleaved) and half-precision qkv/mask inside the
  fused vision SDPA (slower and it drifts). Do not re-adopt without new numbers.

## Learning from oMLX (github.com/jundot/omlx, read at v0.6.4)

oMLX is an MLX inference server for Apple silicon. Its macOS app bundles the full Python
source, which is the reference used here rather than the README, and its published numbers are
measured on an M3 Ultra too, so they compare directly.

- SPECULATION INSIDE A BATCH IS A LOSS, measured there independently: row-wise MTP on
  Qwen3.6-27B / M3 Ultra gives 53.3 / 52.5 aggregate tok/s at batch 2 / 4 against 65.2 / 86.5
  for plain batched decode, DESPITE 83-93% draft acceptance. Their rule generalises ours: a
  drafted cycle carries tokens-per-cycle tokens, so it only beats a batch narrower than that.
  At k=3 this port carries ~2.9, so speculation can only pay below 3 rows.
- AND THAT WINDOW IS EMPTY HERE. `ocr-verify --pdf --batch 32 --report-occupancy` on the 40-page
  scan: 1.4% of decode steps run at <=2 rows and carry 0.2% of the tokens. 62.4% of steps run at
  exactly 14 rows (51.4% of tokens), because the page lengths are 1309/352/73 repeating - the
  short and medium pages finish early and 14 long pages grind together to the end. So
  "speculate once the batch narrows" is worth ~0.3% of a run. Do not build it.
- PREFIX CACHING ACROSS PAGES IS WORTH ~2%, not the large win it is for a chat server. oMLX's
  hot/cold tiered KV cache exists because coding agents resend a long shared prefix; here
  `promptIDs` puts only the chat template's opening ahead of the image tokens, and the ~1000
  image tokens after it are page-specific. There is no long shared prefix to cache.
- THE DECODE STEP IS NOT KV-BANDWIDTH-BOUND, so KV quantization is a MEMORY lever and not a
  speed one. Seeding the same probe with fp32 and fp16 KV at 3072 context: 34.14 vs 34.12 ms at
  32 rows, 21.85 vs 21.83 at 16. Halving the bytes changed nothing. Neither wall is close
  either: at 32 rows / 3072 context a step is 6.04 GFLOP over 6.04 GB of KV, which at 34 ms is
  177 GFLOP/s and 177 GB/s against roughly 27 TFLOP/s and 800 GB/s. Attention at one query per
  row is occupancy-bound, the same conclusion `--probe-gemm` reached for prefill.
- `probeDecodeWidth` seeds fp16 now, which is what `appendStep` actually writes. It seeded fp32
  before and so timed a cache at twice production's size; the numbers are identical either way
  (previous point), which is why the recorded figures still stand.
- CONTEXT COSTS MORE THAN WIDTH DOES. ms/step at 1 / 32 rows is 5.82 / 22.30 at 1024 context and
  6.34 / 34.04 at 3072: one row is nearly flat in context (+9% over 3x), 32 rows is not (+53%).
  A long page in a wide batch is superlinear, which is the mechanism behind the 14-row plateau.
- ANE PREFILL IS REAL, AND `--probe-gemm` MEASURED THE WRONG DOOR. Its conclusion that no ANE
  dispatch materialises is true OF MPSGRAPH; oMLX does not use MPSGraph. It drives a private ANE
  runtime from a native extension, compiling per-layer program banks at a FIXED sequence length
  and splitting each matmul across ANE0, ANE1, the GPU and optionally the CPU. Their M3 Ultra
  figures on Qwen3.8-27B: GPU-only 458 tok/s prefill at 4K, ANE/GPU 588 (+28%), ANE/CPU/GPU 625
  (+36%), fused MLP/down 349 -> 527 (+51%). It holds at q8 (432 -> 557, +29%), which is the
  build we ship, decode is unchanged (prefill-only lever) and peak memory costs about 7 GB.
  Prefill is 14.7 s of a 52.4 s run here, so +30% on prefill would be about +8% end to end.
- THE ANE IS REACHABLE WITHOUT PRIVATE API, AND IT IS FAST, AND IT IS STILL NOT WORTH IT HERE.
  Spiked with plain CoreML (no private runtime): a 1x1 conv over (1, K, M, 1) is the ANE-friendly
  spelling of a linear layer, and at this port's exact prefill shape M1007 K1280 N1280 it runs at
  0.235 ms per GEMM = 14.0 TFLOP/s, against 0.48 ms / 6.8 TF for MPSGraph and 0.72 ms / 4.6 TF for
  MLX. So the ANE is 2.1x the GPU at the shape that matters. Measured by CHAINING L GEMMs and
  taking the slope, with a relu between them - consecutive 1x1 convs collapse into one matmul and
  a fused chain measures nothing. Confirmed from Swift with a preallocated MLMultiArray, which
  also gives the honest fixed cost: ~7.2 ms PER CoreML CALL (7.66 ms at depth 1, 10.00 at 12,
  18.47 at 48). So an offloaded region must be big: one call per page, never one per layer.
- WHAT KILLS IT IS THIS MODEL'S MoE, not the ANE. Per token per layer the GEMM work splits
  attention projections 21.4%, routed experts (top-6 of 64) 67.4%, shared expert 11.2%. Only the
  dense, static-shape parts can go to the ANE, which is 32.6%. The routed two-thirds is a
  top-k gather/scatter; running all 64 experts densely instead costs 10.7x the FLOPs, and at
  2.06x the throughput that is 5.2x SLOWER than top-k on the GPU. So the ceiling is 2.06x on a
  third of the GEMMs, or ~17% of language prefill, which is ~1.9% end to end BEFORE paying for
  fp16 weight copies (oMLX measures ~7 GB, so big-machine only), one compiled program per prompt
  shape, MLX-to-MLMultiArray movement per call, and a CER re-gate for fp16 numerics.
  oMLX gets +28-51% because Qwen3.8's dense MLP/GDN projections are most of ITS prefill.
- "ONLY A DENSE MODEL BENEFITS" IS THE WRONG RULE, and the EMBEDDING model disproves it. It is
  dense (28 layers, h 1024, ffn 3072, GQA 16/8, no experts), pure prefill with no decode,
  already bucket-padded to a fixed set of shapes, and a throughput job - the ideal ANE profile
  on paper. Measured anyway: attention projections 1024->1024 run 26.5 TF on ANE against 14.7
  on GPU (1.8x), but the MLP 1024->3072->1024 runs 10.5 TF against 22.8 (0.48x). The MLP is
  75% of the GEMM work, so the ANE is slow exactly where this model spends its time. It
  plateaus near 11 TF on the wide shape regardless of M, which is what a 3072-channel
  intermediate spilling out of ANE SRAM looks like (~100 MB fp16 at M=16384) - and is
  presumably why oMLX splits the MLP hidden dimension across both ANEs. Offloading only the
  25% it wins caps at 11% of GEMM time and needs one CoreML call per layer, 28 x 7.2 ms.
- THE REAL RULE IS SHAPE, NOT DENSITY. The ANE is roughly flat at 11-26 TF while the GPU swings
  4.6-22.8 TF depending on how well a shape maps to it, so the ANE wins where the GPU is badly
  UTILISED and loses where it is not: 2.1x on the OCR prefill shape (M1007 K1280 N1280, a shape
  MLX runs at only 4.6 TF), 0.48x on the embedding MLP.
- SO oMLX'S WIN IS CONCURRENCY, NOT SPEED. Adding a unit worth ~0.5x the GPU and running both
  at once is worth about their +28%.
- MEASURE THIS STACK IN THIS STACK. A first pass priced the ANE through coremltools and
  MLX-Python and got it badly wrong: coremltools understates the ANE by 4.4x on the MLP shape
  (8.5 it/s against 37.6 from Swift), because the numpy-to-MLMultiArray conversion dominates.
  Every number below is `ocr-verify --probe-ane`, which times MLX-Swift against CoreML natively;
  Python only emits the .mlpackage fixture (`Tools/ane/emit_fixtures.py`).
- THE ANE AND THE GPU ARE PERFECTLY ADDITIVE, measured in-process with CoreML on a background
  queue and MLX-Swift on the main thread: both keep 99-104% of their solo rate at every size
  tried. Do NOT measure this from Python threads - there both collapse to the same rate, which
  is coremltools holding the GIL through the conversion, not memory contention.
- THE SYNTHETIC PROBE OVERESTIMATED BY 4x, AND ONLY BUILDING THE REAL THING SHOWED IT. Chained
  1x1 convs at the tower's shapes put the ANE at 0.57-0.74x of the GPU and perfectly additive,
  which projected roughly +60% on indexing. The ACTUAL 28-layer tower, ported to CoreML and
  verified correct, runs at 3612 tok/s against the GPU path's ~22800: 0.16x. The probe measured
  only the ops the ANE likes, in isolation, with nothing between them.
- AND IT IS NOT ON THE ANE AT ALL. Same compiled tower, L=512, by compute unit:
  cpuOnly 4225 tok/s, cpuAndGPU 15458, cpuAndNE 3676, all 15474. The ANE configuration is SLOWER
  THAN CPU ALONE, and `all` just equals `cpuAndGPU`. CoreML fragments the graph and the transfers
  cost more than the ANE saves. Even CoreML's own GPU path (15458) is well under MLX-Swift's
  22800, so there is nothing here to win with either.
- THE fp32 CASTS WERE NOT THE CAUSE, which was the obvious suspect (57 cast pairs for RMSNorm,
  and casts are known to break ANE residency). A cast-free variant with RMSNorm in fp16 measures
  7.17 it/s against 7.18 - identical - at unchanged accuracy. What fragments the graph is the
  attention block: reshape to (H, D, L), tile for GQA, transpose, batched matmul, softmax. Making
  that ANE-resident needs the full ml-ane-transformers treatment (per-head chunked convs, no
  reshapes), which is a rewrite of the port with no guarantee at the end of it.
- THE PORT ITSELF IS CORRECT AND THE TOOLING IS KEPT, so this is re-testable if CoreML improves:
  `omni-verify dumpbackbone` exports the app's own LoRA-merged fp16 weights (do not re-derive the
  merge elsewhere - that is how you get a wrong answer that looks like a numerics bug),
  `Tools/ane/tower.py` is the MLX reference for the same maths, `Tools/ane/build_tower.py` emits
  the CoreML program and `verify_tower.py` scores it. All eight shipped fixtures pass through the
  CoreML tower at worst cosine 0.99998. Right-padding is free to verify with because the tower is
  causal, so one padded length covers every record.
- SO THE ANE IS CLOSED, on both models and for different reasons: the OCR model because two
  thirds of its GEMM work is a top-k MoE gather the ANE cannot take, and the embedding model
  because its attention will not stay resident. Do not reopen on a synthetic GEMM benchmark;
  reopen only on a measurement of a whole model.
- SPARSE PREFILL (their SpecPrefill) scores prompt tokens with a draft model and prefills only
  the top `keep_pct`. It DROPS prompt tokens, which for transcription means dropping image
  tokens, so it is graded against the CER gate before it is believed, not adopted on its face.
- THE EMBEDDING PATH IS AHEAD, NOT BEHIND, and the transfer runs the other way. oMLX sorts
  inputs by length so a batch does not pad to its longest member, at a fixed batch of 32 with
  `mx.compile` on the forward; `Indexer.embedGroupsReusing` already length-sorts (padding is
  ~0% by construction) and adds a content-keyed vector reuse cache and GPU/CPU double buffering
  that oMLX has no equivalent of. `forwardPooled` also narrows to the pooled row BEFORE the last
  MLP and the final norm, so neither runs on pad positions - oMLX computes the whole
  last_hidden_state and pools afterwards. Most of its embedding code is multi-checkpoint
  plumbing (resolving a pooling mode from sentence-transformers config, remapping input keys)
  that a single-model app does not need.
- ONE CHECK WORTH HAVING RUN: oMLX warns that a bare `[:, -1]` last-token pool is correct only
  under LEFT padding and that an unmasked mean averages pad tokens in, and that both "still look
  fine" on single inputs. This port right-pads. It is correct - `forwardPooled` gathers per row
  with `takeAlong(h, poolIndexGraph(lengths), axis: 1)`, never a bare last column - and the
  backbone being causal means a real token never attends to a pad. Verified, not assumed.

## OCR while the index migrates (Scripts/ocr-during-migration.sh, 2026-09-20)

The two halves of this app that each own a scarce resource, at the same time: the OCR model
saturating the GPU with 4.53 GB of weights resident, and the migration holding the store's serial
queue while it rewrites a 6 GB SQLite file beside a 22 GB vector file. They are supposed to be
independent, and that is the class of claim this file exists to stop asserting without a number.

    OCR solo                    8 pages, 19.7 s, 398 tok/s, digest 8aead74f017d9a6b
    OCR during the migration    8 pages, 19.8 s, 396 tok/s, digest 8aead74f017d9a6b
    the migration meanwhile     48 stamps in 196.9 s, split built, audit clean
    the index afterwards        rowTable=occurrence, 0 failing checks, digest ba7a13400e714f79

Byte-identical transcript, 0.5% on throughput (this machine's run-to-run variance is ~25%), and
the migration took its usual ~197 s. They are independent.

XCUITEST CANNOT DRIVE THE OCR WORKSPACE WHILE A RUN IS IN FLIGHT, and this is the harness, not the
app. Every XCUITest query waits for the app to go idle first, and a run streaming at 24 Hz does
not go idle until it is over - so `ocr.readout`, `ocr.section.0` and `ocr.needsmodel` are all
invisible for the whole run and the app logs `kAXErrorInvalidUIElement ... AXChildren` throughout.
The shipped `OCRWorkspaceUITests` fails this way on a TINY index with no migration at all, twice,
deterministically at exactly the deadline; raising it from 45 s to 240 s changed nothing, which is
what says it is not a timeout. The same build, same fixture, driven by hand with `open -n`,
transcribes in ~25 s in Release AND in Debug, including from inside the runner's own sandbox
container. So those tests now SKIP with that explanation and point at the script above, which is
stronger evidence than a click: it compares a digest.

## OCR weights (published)
- Builds are named by WHAT WAS QUANTIZED, not by a judgement about the trade-off: `q8-mtp-mlx`
  (was "balanced"), `q4-mtp-mlx`, `q8-experts-mtp-mlx`. `Variant.slug` is the one definition; the
  release assets, the install folder and `Tools/ocr/publish.sh` all follow it, and
  `testAssetURLShape` pins it.
- The `q8-mtp-mlx` build is LIVE at github.com/hanxiao/omni-macos/releases/tag/ocr-weights-v1, and
  the in-app download was verified end to end against it: 4.3 GB at ~22 MB/s, model loads, page
  transcribes. `special_tokens_map.json` is NOT published and is not needed - the manifest names
  the three shards, both tokenizer files and `omni-ocr.json`, and that is what the loader reads.
- `gh release upload file#name` sets the asset's LABEL, not its NAME. The asset keeps the file's
  basename, so every URL `assetURL(variant:file:)` builds 404s. `Tools/ocr/publish.sh` now renames
  each asset through the API after upload, which is in place and does not re-send gigabytes.

## Memory between the two models
- THE OCR MODEL ALREADY OFFLOADS. `deactivate()` drops its 4.53 GB on a utility queue when the
  toggle goes off, and `ocrRunActive` stands indexing down for the whole run. Do not re-add
  either as a setting; they are not missing.
- `omniMetalWorkingSetBytes()` is `recommendedMaxWorkingSetSize`, a DEVICE CAPABILITY, not free
  memory. It does not fall when the embedding model loads, so `recommendedWidth` was sizing the
  OCR batch as if 1.8 GB+ of resident embedding weights were not there. `OCRBatchPlan.coresidentBytes`
  is set by the app when the engine loads and enters the reserve. No effect on a machine that caps
  at 32; on a 16 GB laptop it is the difference between a width that fits and one that does not.
- EVICTING THE EMBEDDING MODEL ON THE OCR TOGGLE WAS CONSIDERED AND NOT BUILT. `self.engine = nil`
  frees nothing: `Indexer` holds it as `embedder` and `serving.attach(engine:)` hands it to the
  HTTP/MCP layer, so a real eviction is a bootstrap-level teardown (cancel the indexer, detach
  serving, reload on return) hung off a toolbar click. It buys nothing on a roomy machine, and on
  a small one it trades a multi-second warm-up when you leave OCR for perhaps 15% OCR throughput
  - while risking search and a serving endpoint that promised to answer. Size the batch instead.

## Interrupting the GPU: who yields to whom (audited and measured 2026-09-12)

THERE ARE THREE LANES, NOT TWO, and only one of them is arbitrated.
- `OmniEngine.run(highPriority:)` - a condition-variable gate. Interactive search is high, indexing
  / projection / tagging are low, plus `interactiveQueryActive` (2 s) shrinking the indexing batch
  and splitting a flush into per-batch gate windows. This one is tuned and correct.
- `VectorStore` - 167 MLX operations and NEVER takes that gate. So the scan half of every search
  runs outside the priority system, as does the mask build.
- `OCRModel` - its own MLX lane, no gate. `ocrRunActive` is read in exactly TWO functional places
  (the `startIndexing` guard and a Settings label), so coordination is all-or-nothing: the whole
  indexing pass is cancelled for the whole OCR run. Serving does not know a run is in progress.

VISUALIZATION AND TAGGING DO OCCUPY THE LANE - checked, because it is the obvious thing to get
wrong. The app's projection path wraps every slice in `runLowPriorityGPU` (the ungated
`ProjectionEngine.layout` is the reference path, used only by tests and omni-verify), and the
tagger's score matmul is fused into the image embed inside the gate, with the prior seed an
explicit `run(highPriority: false)`.

THE GATE IS ABOUT SUBMISSION, NOT PREEMPTION. MLX submits through one Metal command queue per
process, so the lanes serialise whatever the gate does; what the gate controls is who submits next
and HOW BIG each submission is. That is why the lever for responsiveness is submission
granularity - which is exactly what `interactiveQueryActive` already does for indexing and what
OCR had no equivalent of.

MEASURED (Release, real index, 21-page PDF, search over HTTP, 59 samples after the cold one):

    search, idle                  p50  14.9 ms   p90  50.8   p99 178.9   first 239 ms
    search, during an OCR run     p50  49.3 ms   p90  51.9   p99  72.5   first 1656 ms

Steady-state contention was real but modest. The damaging number was the FIRST request at 1656 ms:
it lands while OCR prefills a page, which is one large command buffer a query can only queue
behind.

FIXED BY `GPUInteractive` (OmniKit), a process-wide count of interactive requests in flight:

    search, idle                  p50  17.8 ms   p90  71.0   p99 298.0   first   49 ms
    search, during an OCR run     p50  18.1 ms   p90  34.9   p99  36.9   first   37 ms

A search during a transcription is now indistinguishable from an idle one, and the first request
went 1656 -> 37 ms. OCR throughput 505 -> 495 tok/s on the same 21-page document, which is inside
this machine's ~25% run-to-run variance and is anyway a run with no searches in it, so the cost
measured there is only the per-step flag read.

Three things make it work, and each was chosen against an alternative that does not:
- IT IS RAISED AROUND THE WHOLE SEARCH, not just the embed. `run(highPriority:)` raises it BEFORE
  the gate wait, because a query spends most of its latency queued, and `VectorStore.search` raises
  it too - the scan takes no gate at all, so without that the OCR lane resumed submitting the
  moment the embed returned.
- IT IS NOT `interactiveQueryActive`. That flag stays true for 2 s after a query so the indexer can
  keep its batches small. Two seconds is right for choosing a batch size and completely wrong for
  blocking a decode step.
- OCR CONSULTS IT AT THE POLL POINTS IT ALREADY HAD, so no decode code changed: `shouldContinue`
  yields between steps, and `shouldAdmit` refuses to START a row while a query is in flight.
  Admission is where the damage was - it runs a whole page's vision tower and LM prefill.
The yield is ALWAYS BOUNDED (250 ms): the decode loop must keep progressing whatever happens on the
other lane, or a request that never lowers the count stalls a transcription for good.

    stop during steady decode                     98 ms
    stop inside a group prologue, before          2178 ms
    stop inside a group prologue, after            151 ms
    the prologue window (weights up -> 1st token) 4134 ms

THE DECODE LOOP WAS NEVER THE PROBLEM - `shouldContinue` per step honours a stop in 98 ms. The
un-interruptible stretch is the group PROLOGUE: every page of the group rasterised (a 2384 px PDF
render plus the Pillow-exact resample, per page), then `rampRows` pages prefilled at ~650 ms each,
with no cancellation check in either. Both now check, and the A/B above is a real revert-and-rerun,
not an inference.

A STOP DURING MODEL LOAD waits out the load: `OCRModel(modelDir:)` has no cancellation, and the
loop's first `gate.isStopped` check is after it. Bounded (~13 s, once per session) and not fixed.

INSTRUMENT TRAP THAT COST THREE MEASUREMENTS: `open()` calls `reset()` which calls `cancel()`, so
stamping the interrupt time inside `cancel()` unconditionally stamps it AT LAUNCH - and every
"stop latency" it reports is really app-open-to-run-end. It read 28.2 s, 10.4 s and 29.2 s before
the arithmetic gave it up (a keystroke sent at 31 s cannot produce a 29.2 s latency). Stamp only
when `isBusy`.

LEAVING OCR MODE IS NOT A STOP, and it used to produce the identical state: `deactivate()`
cancelled the run and left every unreached page dimmed, to be clicked back one at a time. The
pages still queued are now remembered and re-queued by `activate()`, with `isBusy` as the test so
an explicit Stop is never resurrected. ORDERING MATTERS: `cancel()` clears that debt, so it is
captured before and re-assigned after - written the other way round first, which silently disabled
the whole feature.

A STOP-TRUNCATED PAGE IS NEVER CACHED. `runGroup`'s backstop settles a cancelled row that has
produced text as `.done`, which is right on screen - the reader can see it is half a page - and
wrong on disk, where it would become that file's finished transcript for good. Gated on
`stoppedBy == .cancelled`; `.cap` and `.loopGuard` still cache, being the model's own endings.

PAUSE HAD NO MENU ITEM AND NO KEY EQUIVALENT - it existed only as a button on the floating readout,
which withdraws a few seconds after a run. So the gesture that hands the GPU back POLITELY, keeping
the queue, was the hard one to reach while Stop had Cmd-. It is now File > Pause Transcribing on
Opt-Cmd-. Same argument as the context-menu items promoted in the menu-bar audit.

## Busline efficiency: what is NOT worth doing (measured 2026-09-12)

This round went looking for wasted GPU work on the shared lane and mostly ruled things out. Kept
here so the same ground is not re-dug.

THE OCR GROUP PROLOGUE HAS NO WASTE TO RECLAIM. The suspicion was that rasterising a whole group
up front is seconds of serial CPU with the GPU idle - 21 pages x the 55 ms/page host cost quoted
under the TTFT breakdown. It is 121 ms FOR ALL 21 PAGES (`ocr-group-rasterise`, gated). The 55 ms
figure is `PILResample` inside `preparePage`, which is already deferred per page into the decode
loop; `Self.load` is only the PDF render plus the RGB conversion, ~6 ms a page. So the 3600 ms
prologue is ~3.5 s of genuine ramp prefill. Parallelising the rasterise across cores - which the
TTFT note correctly calls trivially parallel - would buy ~100 ms and needs one PDFDocument per
worker (PDFKit will not render one document concurrently). Not worth it.

THE ROUND-3 INTERRUPT CHECKS COST NOTHING, and this is arithmetic, not a measurement: 25 lock
reads (21 rasterise pages + 4 ramp rows) against 83.6 s of GPU work is ~6 parts per billion, while
this machine's run-to-run variance on an unchanged build is ~25% (401 vs 307 tok/s, recorded
above). An A/B there measures drift. Reference figure for the 21-page arXiv PDF, for whoever wants
a baseline: `ocr-run-done pages=21 tokens=42253 83.6s 505 tok/s`.

A COMPLETED IMAGE EMBED IS DISCARDED ON CANCEL, at the `if self.isCancelled { return }` after
`embedImagesTagged` in `flushImages` - the vision tower has already run and the vectors are thrown
away. NOT CHANGED, for two reasons. It could not be priced (see below), and the obvious fix has a
hazard beside it: a cancel from `setFolderPaused` would then store files under the root just
paused, and `applyIgnoreText` deletes rows from a detached task while a pass may be running, which
is the resurrection shape the tag-backfill note already warns about. A cancel would need a REASON
(pause keeps the work, scope-change discards it) before this is safe - the same distinction the OCR
stand-down needed.

CLOSED, AND THE ANSWER WAS THE PROBE: the full pass's `flushImages` logs nothing on a cold index
because `update()` - the reconcile path - indexes the folder FIRST, through its own cross-file
staging (`iStage` / `flushImagesU`), and the full pass that follows then correctly finds every file
unchanged. Measured on 120 images from a fresh db: eight `image-flush-update` batches of 16 at
~1.0 s each. The batching is alive and there is no batch-1 defect.

Two things led the hunt astray for an hour, both worth knowing. The comment above `update()`'s
staging said "media items keep the per-file path (their batching lives in the encoders)" - stale,
and directly above the code that stages them; it is corrected now. And an instrument that only
covers one of two paths reads exactly like a dead path: the full pass and the reconcile path have
SEPARATE image flushes, and only one was instrumented. `image-flush` and `image-flush-update` now
name both.

SO THE CANCEL DISCARD HAD A PRICE AT LAST: one `flushImages` is ~1.0 s of vision tower work for 16
images, thrown away at the `if self.isCancelled { return }` after the embed - once per cancel, and
a cancel happens on every `beginOCRRun`, folder pause and settings change.

FIXED WITH A CANCEL REASON, which is what it always needed. `cancel(.pause)` keeps work already
done; `cancel()` (the default, and it must stay the default) discards it. The split is not a
judgement call - it falls out of what the call site is doing:
  .pause    pauseIndexing (the OCR run - the most frequent cancel in the app), requestIndexPass,
            indexNewSourcesFirst, startIndexing's re-scope, yieldRetagToSearch
  .discard  setFolderPaused, deleteRowsUnder, root removal, the engine/store swap, quiesceForQuit
The discarding five either SHRINK what the index should contain or tear the store down, and a late
store there writes rows that are about to be, or have just been, deleted - the resurrection shape
`applyIgnoreText` already has to be careful about. `IndexerPauseTests` pins all four cases,
including that the bare `cancel()` defaults to discard: a default of `.pause` would silently make
all five unsafe at once.

RESUMING RE-WALKS THE TREE, AND THAT IS FINE - MEASURED, DO NOT BUILD THE CLEVER VERSION.
`pauseIndexing()` is a cancel, so every resume crawls from the top again. The obvious optimisation
is to replay FSEvents since a stored event id instead (FSWatcher already takes `since:`), and it is
not worth it: the crawl costs 0.24 s FOR 100,000 FILES (`crawl-done`, gated), measured on a
synthetic tree whose files are all below `minTextChars` so nothing embeds - which is precisely the
no-op resume shape, with no consumer backpressure inflating the number. `BulkDirWalker` is
getattrlistbulk-based and linear in entries, so 2.6M files extrapolates to ~6 s of background work
on a utility queue, concurrent with embedding and nowhere near the GPU.

Against that, an FSEvents replay has to be right about DROPPED EVENTS (MustScanSubDirs, UserDropped,
KernelDropped, EventIdsWrapped) or it misses a deletion and leaves stale rows in the index forever -
the same bug class as the Photos ghost rows. Six seconds of background crawling does not buy that
risk. The instrument is kept so the premise can be re-checked if a root ever gets far larger.

## Transcript cache (OmniKit/OCR/OCRCache.swift, 2026-09-12)

A page is transcribed ONCE. Finished Markdown is written to a folder of .md files and read back on
the next open, so re-dropping a document, reopening it next week, or picking up the pages a stop
left behind all cost a file read instead of a GPU minute. On by default, with a switch, a folder
and a Clear button in Settings > OCR.

WHAT IDENTIFIES A TRANSCRIPT: the source file's CONTENT hash, the page index, the prompt, and the
model variant. Content rather than path and date, so a file moved or copied under the same name
hits and a file rewritten in place misses. NOT in the key: batch width and draft length (scheduling
choices that do not change what is decoded - that is the premise of speculative decoding and it is
graded above), and the loop guard, which CAN change a degenerate page. That last one is the honest
gap: a page that only transcribes correctly with the guard off will serve its guarded transcript.
Turning the cache off is the way out, which is what the switch is for.

IDENTITY LIVES IN THE FILE NAME, not in a header inside the file - `Quarterly Report-p7-<16 hex>.md`
- so what is in the file is the transcript and nothing else, and a lookup is one `contentsOfFile`
with no index to keep in step with the directory. The price is that RENAMING a source file misses;
moving or copying it does not. That trade is deliberate: the transcript is the product, and a
directory of 16-hex-digit names is one nobody can use for anything.

THE MEMO NEEDED THE INODE, and the test that was supposed to prove it was passing by luck. Hashing
is memoised per session against size + modification date + inode. Without the inode,
`testEditingTheSourceInPlaceMisses` still passed - but only because `setAttributes` restores a date
a few hundred nanoseconds off; with both versions the same length and the date stamped to an
identical fixed instant, the cache served the PREVIOUS version's transcript. Atomic writes (which
is what `String.write` and every editor here does) always change the inode, so that is the signal.
The test now fails without it and passes with it. What remains is an in-place rewrite, to the same
byte length, with the date forced back, inside one session.

CLEAR ONLY DELETES WHAT IT WROTE. The folder is the user's to choose, so a Clear button that
empties whatever directory is selected is a way to lose a documents folder to one click. Counting
and deleting both go through the same name test (`-p<n>-<16 lowercase hex>.md`), the dialog names
the folder and the count, and a transcript the user has renamed is out of scope for both. Pinned by
`testClearOnlyTouchesItsOwnFiles`.

MEASURED END TO END, two-page PDF, Release build, isolated db and cache dir. Peak RSS over 100 s,
two passes: 8.14 / 9.37 GB with an EMPTY cache (the 4.53 GB of weights load) against 4.49 / 4.98 GB
restoring from a WARM one. A no-OCR launch of the same build peaks at 4.30 GB. The absolutes drift
~0.5 GB between passes - that is MLX's buffer cache, and it is why the no-OCR control is needed -
but the GAP is 3.7-4.4 GB in both, i.e. the weights. A fully cached document never reaches `run()`.
The lookup itself is `ocr-cache-lookup 0.6ms pages=2 hits=2` (gated on OMNI_PERF_LOG).

TWO TRAPS IN MEASURING THAT, both of which produced a confident wrong answer first:
- `-omni.ocr.cache.enabled NO` DID NOT DISABLE IT. Launch arguments land in the argument domain as
  STRINGS, so `object(forKey:) as? Bool` casts to nil and falls back to the default - the control
  arm ran with the cache ON and the A/B compared two arms of the same thing. `isEnabled` now reads
  "unset means on" then `bool(forKey:)`, which coerces both spellings. To disable it in a test,
  point it at an EMPTY DIRECTORY.
- PEAK RSS IS NOT READABLE WITHOUT THE NO-OCR CONTROL. A single 4.50 GB sample was read here as
  "the model loaded"; the baseline with no OCR at all is 4.30 GB, so it meant nothing. Only the
  8.14 vs 4.49 split against that baseline says anything.

THE LOOKUP GATES THE RUN, it does not race it. Pages arriving in a drop go into `awaitingCache` and
the run loop skips them, so a page about to be restored is not decoded in the moment before the
lookup lands; `applyCacheHits` empties that set and then starts the run for whatever actually
missed. Two consequences worth knowing: a fully cached document never reaches `run()` at all, and a
run already winding down gets a second `startRunIfNeeded` 400 ms later, because it may have taken
its last look at the queue while those pages were still held.

EXAMPLES, NOT PRESETS, in the prompt box. The rule above ("presets change the SHAPE of the output,
which is not something a reader can judge from a preset's name") stands and is why these are whole
prompts that land IN the editable box, where they can be read and changed before they are used -
not a picker that swaps the prompt from behind a label. Each is a complete prompt, because a
fragment appended to the default contradicts rules the default has already given.

## OCR settings
- ONE build is offered, and the weights live at `Application Support/Omni/jina-ocr-v1-<slug>` beside
  the embedding model, not in an `ocr/` of their own. `OCRModelCatalog.migrateLegacyInstall()`
  renames an older install once, at LAUNCH - not when Settings opens, because the workspace asks
  whether the model is installed long before anyone visits a tab.
- The build picker, its size/throughput/CER line and the speculative-decoding controls are GONE.
  Those are numbers nobody outside this repository can act on, offering a choice whose wrong
  answers are measurably worse (4-bit scores CER 0.25 on handwriting) - the app picks. The k curve
  and the variant measurements stay recorded above; they belong here, not in a settings pane.
- There is NO Remove button for the OCR model. Deleting four gigabytes is something a person does
  where they can see what they are deleting; `Application Support/Omni` is watched with a debounced
  `DispatchSource`, so the settings row is right whether the folder goes from the Finder or not.
  The watch handler MUST be built in a `nonisolated static` helper: a closure written inside a
  `@MainActor` method inherits that isolation and the runtime traps the first time it fires on the
  dispatch queue (EXC_BREAKPOINT in `swift_task_isCurrentExecutor`).
- The prompt is one editable box. Presets that drop the LaTeX, HTML-table and header/footer rules
  change the SHAPE of the output, which is not something a reader can judge from a preset's name.

## OCR over HTTP and MCP (issue #22, 2026-09-22)

Three surfaces, one core (`App/Serving/OCRServing.swift`): `POST /v1/chat/completions` (OpenAI
shape, SSE streaming), `POST /v1/ocr` (Mistral OCR shape, 0-based `pages`), MCP `ocr` (1-based
`pages`, 10 per call, paths only). Text parts are ignored: the model does not act on prompts.
- ONE MODEL, ONE DECODE. `OCRModelHost` leases the weights to the workspace and to served
  requests (served lease lingers 120 s) and owns a FIFO decode slot. A served request never queues
  behind the workspace: 503 + `Retry-After` in ~2 ms. A served request after a workspace run
  reuses the loaded model (1.3 s for one image, no reload).
- NEVER AWAIT THE MAIN ACTOR ON THE SERVED PATH. A folder-access prompt on a fresh build held the
  main thread in `open()` and a served request hung behind `MainActor.run` for 11 minutes. Hooks
  into AppModel are posted to the main queue (FIFO), never awaited.
- THE LIVE TAIL IS HELD BACK (`OCROrderedText.holdback`). A streamed update can carry a token past
  the page end that the final result drops (1 page in 40 streamed a trailing "skap"). With 32 bytes
  held back, 40 freshly decoded pages stream byte-identical to the non-streamed answer.
- A JOB THAT TOOK THE SLOT MUST RUN. `streamBody` calls the producer even when the head fails to
  send, and `Job.deinit` returns an unrun slot; otherwise OCR and indexing stall until restart.
- Batched decoding is not bit-identical across batch compositions: two runs of the same 40 pages
  (37 vs 40 fresh) differ by one character on one page. The workspace behaves the same.
- Served decode footprint matches the workspace: the 0.13.6 control reads 22-23 GB during and
  48 GB after a 40-page run (MLX buffer cache with the cap lifted); a served 10-page request reads
  24 GB and returns to 2.6 GB when the linger ends.
- QUIT EVERY ISOLATED INSTANCE WHEN ITS CHECK IS DONE, and look with `pgrep -x Omni` before calling
  anything finished. Four were found still running after two days: they held GPU and memory, and
  the two with clipboard capture on had recorded the user's clipboard into scratch folders.
- Test the server ISOLATED: `-omni.dbDir`, `-omni.addedFolders`/`-omni.roots` on a scratch corpus,
  `-omni.ephemeralUIState YES`, `-omni.serving.port 51399`, `-omni.ocr.cache.dir <scratch>`. A dev
  build on the real roots raises folder prompts that must not be answered for the user.

Serving review fixes in the same change: every embedding schema honours the query/document role
and refuses an unknown one (query vs document cosine for one text: 0.876); a Gemini batch embeds
each row by its own `taskType`; malformed batch items are 400s, not dropped rows; `/v1/search`
normalizes folders and kinds; the idle timer no longer closes a connection while its handler works;
upload buffering is in place (was quadratic), answers `100-continue`, and allows 48 MB only on the
OCR routes for an authorized caller; symlinks cannot lead a path argument out of the indexed folders.

SKILL.md (`ServingTab.skillMarkdown`) is reference, not manners: endpoints, fields, limits, errors.
It is for agents that call the HTTP API, so it NEVER describes MCP - no MCP section, no "over MCP"
asides. MCP clients get their docs from the tool descriptors and the initialize instructions.

## OCR memory, first page, closed tabs (2026-09-22)
- A 200-page scan held 179 GB after the run (188 GB peak) on 0.13.8. Two causes, both measured.
  (1) OCR mode set MLX's buffer cache to physical/3 (170 GB here), and batched decode churns it:
  every step's attention and mask are sized by a cursor that moves. `omniSetOCRMemory` bounds it at
  physical/16 in [1, 16] GB, and `endOCRRun` clears it. The ceiling was 4 GB first, which cost 8%
  IN THE APP (346-351 tok/s against 374-386 for 0.13.8, installed and source-built alike) though
  only 1% headless: the app allocates beside the run. At 16: 371, peak footprint 28 GB against 46. 40 pages at width 32: 379 tok/s with no
  cache, 445 at 1 GB, 451 at 2, 462 at 4, 466 at 170; 200 pages 675 at 4 against 678 at 170.
  (2) the continuous batch's shared KV cursor counts the steps of the whole RUN, so the buffer
  followed the document. `OCRBatchKVCache.compact` packs each row's live history under the
  longest row instead of growing (keys are rotated by logical position, so moving them is free).
  200 pages: MLX peak 25.0 -> 16.5 GB, 636 -> 678 tok/s, digest 69c75ca3faff9f0d unchanged.
  In the app, back to back: footprint 15-20 GB for the whole run and 7.6 GB after, against
  23 -> 188 GB and 179 GB after.
- `decodeContinuous` held every admitted page's pixels and visual features until the document
  returned; it now keeps only ids and grid after admission. Pages are rendered when admitted
  (`transcribeBatched(sources:)`, `PreparedPage.source`), not all before the first token; a page
  whose source returns nil finishes as `.unreadable` and the batch carries on.
- RAMP PACING (`OCRRuntimeFlags.rampDecodeShare`, 0.25): while the batch widens, decode that share
  of the last admission's time before admitting the next page. The pure one-admission-per-step
  ramp is almost all prefill, so no page finishes until every row is in. Numbers are in the flag's
  comment; 0.25 moves the first finished page 15.3 -> 12.1 s in the app (re-measured 14.9 -> 12.0 s
  with the 16 GB cache: 371 against 366 tok/s, noise), 0.5
  costs 4-10% there. hard2 CER is identical at every share. A narrow batch (width <= 4) starts
  full and never paces, so grade it at width 10 or the gate measures nothing.
- Closing a tab mid-run drops its pages from the batch (`shouldDrop`, `OCRRunGate.drop`): a running
  row is freed at the next step and a queued page is never rendered. `ocr-verify --grade-batch
  --drop-after S` checks it headlessly: 24.2 s -> 13.1 s, the kept pages unchanged.
- An edited tab shows pages finished after the edit (`editBaseByDocument`), after its own text.
- Do not drive the dev app with cliclick while Han's terminal is full-screen: the dev window is
  behind it and the clicks land in his terminal. `screencapture -l` captures a covered window, so
  a screenshot does not show that it is covered.

## MLX 0.32: measured, held back (2026-10-03)

mlx-swift is pinned `exact: "0.31.3"` in Package.swift. 0.32.3 (core 0.32.2, needs Swift 6.3,
i.e. Xcode 26.6 at /Applications/Xcode-26.6.0.app via DEVELOPER_DIR) was built and graded end to
end: embeddings and search unchanged (fixtures 0.99992 worst, search digest moved only by exact
ties), OCR CER on hard2 8/10 exact against 7/10, the M=2/3 qmm bug fixed upstream (`--probe-qmm`
all OK). It was NOT taken because it makes the OCR decode step ~4 ms slower at every width:
5.7 -> 9.8 ms at 1 row, 33.7 -> 37.9 at 32, so the single-page path drops 194 -> 155 tok/s.

- THE COST IS MLX's ROUTED-EXPERT GATHER, isolated by removing parts of the real step under both
  versions: without attention or the LM head the gap stays; without the MoE it is gone (3.39 vs
  3.48 ms); without the routed experts it is gone; without the shared expert it stays.
  `ocr-verify x --probe-gather` reproduces it with no model: 12 chained MoE layers, router computed
  in the graph, 1.67 ms on 0.31.1 against 5.6-5.9 ms on 0.32.2 (13.0 against 17.5 at 32 rows);
  2.15 vs 1.15 even with constant indices. Each op ALONE times the same on both versions, so it
  is latency in a dependent chain, not kernel throughput. Re-run that probe on every MLX release
  and move the pin only when it matches.
- RULED OUT, each measured: the command-buffer caps (MLX_MAX_OPS/MB_PER_BUFFER up to 1,000,000,
  including oMLX's 200/512 - no change or worse), transposed expert weights (same +4 ms), the new
  write-after-read barrier in `register_output_array` (removed in a local build: no change), the
  JIT/metallib setup, and host-side encoding (the main thread is ~70% in the GPU completion wait in
  both). The Metal System Trace shows the new build splitting a command buffer into up to five
  compute encoders and the GPU 67% busy against 86%. Root cause inside MLX not found.
- SWIFT 6.3 CHANGES OCR OUTPUT BY ITSELF, independent of MLX: it merges `cosf`/`sinf` on one
  argument into `__sincosf_stret`, which differs by up to one ULP, and the rope table moved the
  40-page digest 772db0f94e0ae106 -> a35ef0f9c8fe8ad (3 chars). Same source with Swift 6.2 gives
  the baseline. `ropeCos`/`ropeSin` are `@inline(never)` so the pair is never seen; with them the
  Swift 6.3 build gives 772db0f94e0ae106 (single, 196 tok/s) and b16f69c903bb21cd (width 32, 460),
  both identical to the 6.2 baseline. ANY new toolchain: re-run the digest before trusting it.
  (The folder map's UMAP rotation has the same pair; it only moves a layout, so it was left.)

THE CAP CAME BACK UNDER A RUNNING OCR (fixed 2026-10-03). `bootstrap()` and `loadPerf()` applied
the user's 6 GB cap with no regard for an OCR hold, so a launch straight into OCR mode (or a
relaunch restoring it) ran the whole transcription under the cap: the decode thread sampled at
~75% in MLX's over-the-limit backpressure (`get_active_memory`/`get_memory_limit` around
`scheduler::wait_for_one`, under mutex traffic). It is a RACE, which is why it hid: the same source
(9252fdc) built with Xcode 26.2 ran the 40 pages in the app in 57.0 s (418 tok/s), built with
26.6 in 136.9 s (174). `applyMemoryLimit` now keeps OCR's settings while OCR holds the memory.
In the app, isolated, window not raised, back to back: shipped 0.14.5 73.9 s / 322 tok/s, first
page 20.5 s; HEAD with the fix 47.5 s / 501 tok/s, first page 12.5 s. `ocr-first-token` logs
the live limit now (OMNI_PERF_LOG) so this can be read rather than inferred.
- A STOP DURING MODEL LOAD (the "~13 s, not fixed" note above) is obsolete: a cold load from a
  fresh APFS clone of the weights is 1.1 s, warm 0.6 s, the load-time re-grouping included.

TWO MORE, 2026-10-03, after validating the old notes' premises:

- THE NEXT PAGE'S CPU HALF IS PREFETCHED in continuous batching (`preparePageHost` on a background
  queue, one page ahead; `finishPage` runs only the tower inline). Admission used to render, hash
  and resample inline, ~56 ms a page with the GPU idle. 40 pages at width 32: 43.2 -> 40.05 s,
  552 -> 595 tok/s; hard2 width 10 unchanged (8/10, 0.0086), first page 7.4 -> 6.9 s. The earlier
  "host-only prefetch is worth 0.2-0.9 s" was measured on the static pipeline, before continuous
  batching put admissions inside the decode loop. The batched digest can move run to run: ramp
  pacing admits by elapsed time, so faster admission can change which pages share a step.
- ADAPTIVE DRAFT LENGTH: NOT BUILT, now on its merits. Exact verify removed the output objection,
  but a perfect per-page oracle over k = 2..6 on hard2 gains only 1.9% over k = 3 (per-page bests
  range k = 2..5, at most +6.7% on one page), and k = 3/4/5 run within 1% on the 40-page scan. A
  real controller would pay exploration on top.

THREE THINGS TAKEN FROM oMLX (read at 5dcfe24), all on 0.31.3, all measured back to back on the
40-page scan and graded on hard2 against the torch oracle:

- EAGER PER-LAYER DISPATCH (`OCRRuntime.eagerMaxRows`): decode-shaped forwards `asyncEval` each
  decoder layer as it is built, so the GPU runs layer i while the host encodes i+1. Scheduling only,
  digests unchanged: 195 -> 201 tok/s single, 460 -> 467 at width 32.
- EVERY PACK IS RE-GROUPED AT LOAD to the `transpose: true` layout (`OCRWeights.kMajor`), whose MLX
  kernels are faster. A SECOND ROUNDING, so it was graded as one: hard2 page for page unchanged
  (greedy 7/10 exact, CER 0.0087, every divergence at the same character; width 10: 8/10, 0.0086).
  201 -> 234 tok/s single, 465 -> 552 at width 32, first token ~655 -> ~540 ms, load still 0.5 s.
  `transpose: true` has no M=2/3 bug (`--probe-qmm` prints both), so `safeQuantizedMM` pads only an
  untransposed pack - which a loaded model no longer has.
- EXACT VERIFY ROWS: speculative output is byte-identical to greedy at every k (hard2 10/10 pages
  at k = 2..6; 40-page digest a35ef0f9c8fe8ad at greedy and k = 2..5), at no cost (k = 3 decodes
  274.9 -> 276.6 tok/s on hard2, 234 on the 40-page scan). Three sources of row-count dependence,
  each found with `ocr-verify x --probe-rowexact [--causal-sweep]`:
  - quantized projections: in the transpose:true layout MLX runs a few rows through its one-row
    qmv kernel row by row, so they were already exact once the packs were re-grouped.
  - plain bf16/fp32 matmuls (attention projections, router, LM head): one row uses gemv, several a
    tiled gemm that IS row-invariant for M = 2..8. `ocrProjRowInvariant` pads a one-row target step
    to two, so greedy shares verify's kernel. The draft head opts out; its output is only a guess.
    Reshaping rows to (n, 1, K) does NOT help: MLX folds the batch back into M.
  - attention: MLX's one-query kernel plans one or two passes, and a two-pass block count, from the
    call's total key count and its query rows - one causal call over k+1 rows straddling 1024 keys
    (every ~1007-token page prompt does, a few tokens in) planned some rows differently from their
    own greedy call. `OCRVerifyAttention` groups rows by plan, mirroring MLX's rule, capped at 8
    query rows (past that MLX leaves the one-query kernel); `selfCheck` at model load compares
    grouped against per-row calls across every switch and falls back to one call per row if the
    rule disagrees. One call per row was the first version and cost 7% at k = 3.
  The 40-page digests are now a35ef0f9c8fe8ad (single, every k) and f1ed744e12ae453 (width 32).
  Batched decode at wide widths still differs from single: past MLX's qmv row limit the quantized
  projections use qmm. `ocr-verify --pdf --draft 0` has always meant one token a page (the
  document path has no greedy branch for 0); greedy is `--draft 1`.
