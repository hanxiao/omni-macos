# jina-ocr-v1 on MLX-Swift

An in-process port of `jinaai/jina-ocr-v1` (DeepSeek-OCR architecture: SAM-ViT-B + CLIP-L
DeepEncoder, DeepSeek-V2 MoE decoder, FastMTP draft head) running on the Mac GPU through
MLX-Swift. No Python at runtime, no server.

It is an optional add-on. It is not downloaded unless asked for and is not on the indexing or
search path.

## Layout

| file | what |
|---|---|
| `Sources/OmniKit/OCR/OCRWeights.swift` | sharded safetensors + per-tensor quant packs |
| `Sources/OmniKit/OCR/OCRPreprocess.swift` | Pillow-exact resampling, crop/pad layout |
| `Sources/OmniKit/OCR/OCRResize.swift` | ATen antialias bicubic for the positional tables |
| `Sources/OmniKit/OCR/OCRVision.swift` | SAM + CLIP + projector |
| `Sources/OmniKit/OCR/OCRLanguage.swift` | MoE decoder, KV cache, MTP head |
| `Sources/OmniKit/OCR/OCRModel.swift` | prompt, visual assembly, generation |
| `Sources/OmniKit/OCR/OCRSpeculative.swift` | FastMTP draft/verify decoding |
| `Sources/ocr-verify/` | the numeric gate and benchmark |
| `Tools/ocr/make_hard_pages.py` | the hard-page corpus generator |
| `Tools/ocr/ref_dump.py` | torch reference dumps |
| `Tools/ocr/convert.py` | HF checkpoint -> sharded, dynamically quantized MLX build |

## The oracle

The reference is the **original HuggingFace checkpoint at its shipped bfloat16 precision**, run
under torch/MPS. Not the MLX python port, not an fp32 upcast.

Two levels of evidence, answering different questions:

* **stages** - every intermediate against torch's own tensor. Localises a defect to one op. The
  only level that can see a bug the output happens to survive.
* **greedy** - the complete transcription, character for character, on pages the reference itself
  finished at EOS. The only level that can see a bug the tensors survive (a detokenizer that
  corrupts split multi-byte glyphs passes every tensor check ever written).

A third number makes the other two readable: the **horizon**, the first character where torch's
own bf16 and fp32 runs disagree. Past it there is no canonical text, so a difference there is the
dtype's and not the port's. On `doc.png` the horizon is 52% of the page; on `sweep_8pt` it is 4%.
Any "exact" claim has to state the compared range.

```
swift build -c release --product ocr-verify
.build/release/ocr-verify <modelDir> <refRoot> --horizon <fp32RefRoot>
```

## Fidelity

Seven pages, every one generated to its natural EOS (complete documents, not prefixes), graded
against the torch bf16 reference. `CER` is Levenshtein distance over reference length.

Two corpora. The **easy** set is seven clean synthetic pages; the **hard** set is ten pages built
by `make_hard_pages.py` - handwriting, a ruled financial grid, a thermal receipt, a boxed form,
7pt two-column body text, each also in a damaged variant (JPEG 18-30, blur, sensor noise, skew,
illumination gradient, reverse-side bleed-through, perspective keystone). The hard set is the one
that discriminates: on the easy set every build agrees, which is exactly why it cannot be trusted
alone.

Decode figures include FastMTP speculation (k=3, on by default), measured in isolation, two runs
per build, spread under 1%.

| build | size | hard exact | hard CER | easy CER | decode | vs torch |
|---|---|---|---|---|---|---|
| torch bf16, MPS (reference) | 7.4 GB | - | - | - | 38.3 tok/s | 1.0x |
| bf16, no quantization | 6.67 GB | 8/10 | 0.0039 | 0.0231 | 160 tok/s | 4.2x |
| **fidelity** (`q8`) | 4.62 GB | 8/10 | 0.0082 | 0.0159 | 198 tok/s | 5.2x |
| **balanced** (`dyn-k`) | 4.53 GB | **9/10** | **0.0044** | 0.0285 | **205 tok/s** | **5.4x** |
| **compact** (`dyn-j`, dynamic 4-bit) | 4.13 GB | 2/10 | 0.0465 | 0.0469 | 203 tok/s | 5.3x |

`balanced` is the recommended build on the evidence: it matches the unquantized model's accuracy
on hard pages (CER 0.0044 vs 0.0039) at a third less size and 28% more throughput.

**The hard corpus changed the answer.** On the easy set `compact` looked like a reasonable
trade - CER 0.047 against 0.032. On the hard set it collapses where it matters: CER **0.25 on a
clean handwritten page** every other build transcribes exactly. Handwriting is the content 4-bit
routed experts cannot hold, and no amount of clean printed text would ever have shown that.

TTFT on M3 Ultra: 191-233 ms single-view, 440-750 ms multi-crop; the quantized builds pay ~25%
more TTFT than bf16 because the fused MoE prefill is slower on packs, and win it back several
times over in decode.

`fidelity` scoring a lower CER than unquantized bf16 is not a claim that quantization improves
the model. The two differ only in which way a handful of greedy near-ties fall, and both are
3/7 exact; they are the same quality point within tie noise.

Prompt token ids are identical to the reference on every case. Preprocessed pixels
(`images_ori`, `images_crop`) are **byte-identical** to torch's.

`3/7 exact` is not `4/7 broken`. Prefix-exactness and transcription quality disagree sharply and
both are reported for that reason: the compact build's `doc_small` "diverges at char 482 of 690"
and the entire difference is `class="layer-row"` against `class="layers-row"` - one letter inside
an HTML attribute that never renders, CER 0.0014.

## Where the remaining divergence comes from

Measured, not inferred.

**One token in 1007 routes to a different expert.** On `doc.png`, layers 0-7 agree with torch to
fp32 rounding (rel ~1e-7) and layer 8 jumps 100x. The row-level breakdown says exactly 1 row of
1007 moved, and dumping torch's own `topk` indices confirms it: 11 of 12 MoE layers select
byte-identical expert sets, and one token at layer 8 does not. Two router probabilities within
fp32 rounding of each other, resolved differently by `torch.topk` and MLX's `argSort`. This is
not fixable by precision - it is a discrete choice made on equal values.

**`doc_math` is the one page where the port diverges before torch's own horizon** (char 78 of a
fully determinate 1312). The fp32 build moves it to 278, so it is precision-driven rather than a
logic defect, and the divergence is a formatting mode flip: the reference emits
`$A = \text{softmax}(...)$` and the port emits the same content without the LaTeX wrapper. Every
build in the ladder, including unquantized bf16, flips at exactly this character.

## Speculative decoding (FastMTP)

The draft head is one transformer block, ~70 MB, and it is included in every shipped variant. It
drafts k tokens; the target verifies all of them in a single forward; the accepted prefix commits.

Draft-length sweep, mean decode over three pages against 183 tok/s greedy:

| k | 1 | 2 | **3** | 4 | 5 | 6 | 8 |
|---|---|---|---|---|---|---|---|
| tok/s | 173 | 196 | **200** | 199 | 190 | 183 | 148 |
| acceptance | 0.82 | 0.69 | 0.58 | 0.51 | 0.46 | 0.41 | 0.32 |

k=3 is the peak. k=1 is a net LOSS - one draft cannot pay for its own forward. Acceptance at
draft position 0 is 0.79-0.89, which is the healthy FastMTP regime; the earlier python port sat
at 0.12-0.20 and concluded speculation "does not pay off on this stack". It was right about its
own numbers and wrong about the model: the defect was an off-by-one in what the head consumes.

**The contract, from the reference.** At position j the head takes the token AT j paired with the
target hidden from `j - 1` (EAGLE-style), returns POST-norm hidden, and feeds that same tensor
back as the next step's `previous_hidden_states`. Pairing token j with hidden j instead - which
is what the python port did - has identical shapes, produces correct output (the target verifies
everything), and quietly costs 4x acceptance. Nothing but a measurement can catch it.

**Not bit-identical to greedy, and the difference is characterised.** A verify pass computes k+1
logits at once with a different reduction order than a one-token forward, so a near-tie argmax can
fall the other way. Measured over 17 pages: 14 token-identical, 3 differ by exactly one token,
deterministically. Aggregate quality does not regress - mean CER is equal or better with
speculation on both corpora (0.0087 -> 0.0044 on the shipped build).

**The checkpoint's "self-contained" draft tensors are duplicates.** `mtp_embed_tokens`,
`shared_head.local_head` and `shared_head.norm` are bit-identical to `embed_tokens`, `lm_head` and
`norm` (max|d| = 0.000e+00 on all three), and building with or without them gives the same
acceptance to the individual count. The converter omits them; the artifact is 662 MB smaller.

## Long documents

A document is **N independent single-page requests**, not one request with N images. The model
accepts several `<image>` markers and doing so is a trap: the chat template places them adjacent
with no delimiter, and the model transcribes the LAST one. The predecessor established this
against torch - identical N-image path in PyTorch, byte-identical output from both, in both page
orders - so it is the model's behaviour, not a port artefact.

`OCRModel.transcribe(pdfAt:)` rasterises one page at a time through `FileExtractor.renderPDFPage`,
so peak memory is one page of pixels plus the model regardless of document length.

Measured on a 40-page synthetic scanned PDF (`Tools/ocr/make_long_pdf.py`: ruled ledger tables,
prose, cursive field notes, each JPEG-compressed, blurred, noised, skewed and shadowed):

| | 40 pages | per page | aggregate |
|---|---|---|---|
| balanced, workers=2 | **132.9 s** | 3.32 s | 179 tok/s |

All 40 page markers recovered in place; page 1's 22-row table came back with its `PAGE TOTAL`
(59899.00) exactly right.

**Parallelism does not pay in-process.** Three overlap strategies, 9-12 pages each:

| strategy | time | vs sequential |
|---|---|---|
| sequential | 30.6 s | - |
| host prefetch (rasterise+resample a page ahead) | 30.5 s | +0.3% |
| vision prefetch (also run the vision tower ahead, own MLX stream) | 30.0 s | +2% |
| 2 concurrent page lanes, own MLX streams | 38.8 s / 12p | +5% |
| 3, 4, 6 lanes | 38.8-40.0 s / 12p | no further gain |

Concurrency saturates at 2 lanes for +5%. That is far short of the **1.8x the predecessor measured
with 6 worker PROCESSES**, and the gap is the interesting part: separate processes each get their
own Metal command queue, while streams inside one process still serialise on submission. So
multi-process is the only route to real page parallelism here, and it costs N x 4.5 GB of weights
- which is why it is not the default. `transcribeConcurrent(workers:)` exists, defaults to 1, and
is byte-identical to sequential (same document digest at 1, 2 and 4 lanes).

**Output length is derived, not guessed.** The old fixed 1024-token cap silently truncated a fifth
of every ledger page - they need 1309 tokens and stopped with `stopped_by = cap`. `OCRTokenBudget`
now computes the cap from the model's 32768-token context window and this machine's Metal working
set: 65.0 KB of KV per token, so the full window costs ~2.1 GB and every Mac that can hold the
4.5 GB model is context-bound rather than memory-bound. On this machine that is **31753 tokens per
page** instead of 1024. Runaway output is the loop guard's job, not the cap's, and the guard costs
nothing measurable (175 vs 175 tok/s with it on and off).

Raising the cap exposed an O(n^2) term: the KV cache grew by fixed 256-token blocks and each
growth concatenated the whole cache, which at 30k tokens is ~115 GB of copying across 117
reallocations. It now doubles.

**Exact vs torch is not the same as correct.** On the cursive field-note pages the port matches
torch exactly (CER 0.0000) and torch reads "depth 7.8 metres" where the page says 4.3, and
"G-008-69" where it says C-003-69. The port is faithful; the model cannot read this script face.
Every fidelity number in this document measures the former.

## An MLX bug this work uncovered

`quantizedMM(x, w, transpose: false)` in mlx-swift 0.31.3 returns unrelated numbers - relative
error ~1.5, not a precision loss - when the row count is exactly 2 or 3. M=1 and M>=4 are correct
to ~1e-6, at 4 and 8 bits and group size 32 and 64 alike. Reproduce with `ocr-verify --probe-qmm`:

```
bits=4 gs=64:  M1 2.4e-07  M2 1.2e+00!  M3 1.5e+00!  M4 1.3e-06 ... M8 1.2e-06
```

Nothing shipped before this was affected - greedy decode runs at M=1 and prefill at M in the
hundreds. It surfaced only when speculative verification started forwarding k+1 tokens, where
k=1 and k=2 land exactly on the broken widths, and it surfaced as plausible wrong tokens rather
than an error. `safeQuantizedMM` pads the row dimension to 4 and slices back;
`OCRPortTests.testQuantizedMatmulIsCorrectAtEveryBatchWidth` pins it.

Worth reporting upstream: the repro is model-free and three lines long.

## The quantization ladder

`Tools/ocr/convert.py --policy <name>` builds any row. Bit width is chosen **per tensor role**
and recorded in `quant_map` metadata: a single global `(bits, group_size)` cannot describe a
mixed file, and a quantized matmul handed the wrong group size returns plausible garbage rather
than an error.

Single-factor results, all against the same seven pages:

| change from the 8-bit baseline (`q8`) | decode | quality cost |
|---|---|---|
| shared expert + dense MLP -> 8-bit | **+12%** | `sweep_8pt` CER 0.019 -> 0.107 |
| attention -> 4-bit gs32 | +10% | `doc_small` 690 -> 482, `doc` -> 382 |
| routed `down` -> 4-bit gs32, `gate_up` wide | +12% | none on `scan_unique`, `doc_dense` |
| routed `gate_up` -> 4-bit gs32, `down` wide | +13% | `scan_unique` 1231 -> 27 |
| routed experts, late layers only -> 4-bit gs32 | +0.3% | `scan_unique` 1231 -> 27 |

Two findings shape the shipped builds.

**The speed was not coming from 4 bits.** The largest single win in the table is quantizing the
SHARED expert and the layer-0 dense MLP to 8 bits - tensors every token passes through, unlike the
routed experts where only 6 of 64 are read. `balanced` contains no 4-bit tensor at all and is
within 1% of the fully dynamic 4-bit build's throughput. Quantizing the routed experts further
buys 9% of download size, not speed: decode here is bound by fixed per-launch latency at these
skinny shapes, not by weight bytes.

**Inside the routed stack, `down` tolerates 4 bits and `gate_up` does not** - same speed either
way (170.5 vs 171.2 tok/s), so the split is chosen purely on accuracy. That is what `compact`
does with its 4-bit budget.

Rejected, with the reason recorded so they are not re-derived:

* **Uniform 4-bit** (`q4`, `dyn-g`): breaks `doc_dense` at char 1301 and loses the ability to
  stop - `doc_multiling` runs to the token cap instead of EOS, CER 2.36. A build that cannot
  terminate needs an external stopping rule, which is its own failure mode.
* **`dyn-f`** (all routed experts 4-bit, attention wide): same runaway on `doc_multiling`.
* **Keeping the shared expert wide to rescue 4-bit routed experts**: does not work. A build
  differing from plain 4-bit only in shared-expert precision diverges at the identical character.
* **Uniform 4-bit on handwriting**: `compact` scores CER 0.25 on a clean handwritten page. Do not
  ship a 4-bit routed-expert build for anything but printed text.

Quantization is not a free speed lever on this model. `fidelity` is only 1% faster than
unquantized bf16 while being a third smaller, and under the earlier unfused MoE dispatch it was
5% SLOWER. A narrower kernel does not automatically run faster.

## What actually moved decode

| change | effect | exactness |
|---|---|---|
| fused gather-matmul MoE dispatch (`gatherQuantizedMM`) | **+17% decode** | unchanged |
| fused SDPA in SAM (rel-pos bias as additive mask) | 1.42x on the vision tower | bit-identical |
| `MLXFast.rmsNorm` / `layerNorm` | 1.35-4.6x on their shapes | ~1e-6 |

The MoE dispatch is the largest win. Measured on `doc_dense` (766 output tokens, one process per
setting):

| MoE dispatch | decode | TTFT |
|---|---|---|
| never fused (grouped) | 138.9 tok/s | 666 ms |
| fused at n <= 16 | 152.3 tok/s | 665 ms |
| always fused (shipped) | 162.1 tok/s | 736 ms |

Fused computes `n * topK` expert rows where grouped computes `active * busiest`, so fusing the
~1000-token prefill costs ~70 ms of TTFT and buys 6.4% of decode; break-even is ~180 output
tokens, below every page in the reference set. `OMNI_OCR_FUSED_MOE` sets the crossover.

One thing is deliberately not claimed: why the PREFILL dispatch changes DECODE throughput at all,
when both settings decode at n = 1 through the same code. Allocator pool state is the obvious
suspect and it has not been measured.

Not adopted, measured:

* **`MLXFast.rope`** - 1.24-1.44x at exactly our shapes and numerically wrong here. The
  checkpoint's rotary is Llama split-half with cos/sin duplicated across halves; MLX's
  non-traditional path is interleaved and its traditional path pairs `(i, i + dims/2)`. A drop-in
  corrupts every attention score by ~1.4e+01 against values of order 8 while reporting a speedup.
* **Half-precision qkv or mask inside the fused vision SDPA** - 15-80% slower (the casts over the
  huge operands cost more than the narrower matmul saves) and it drifts.

## Preprocessing

The reference processor resizes with `PIL.Image.resize`, whose 8-bit path is fixed-point:
coefficients quantized to 22 fractional bits, accumulated in int32 with a rounding term, clipped
to uint8 after each of the two separable passes. `PILResample` reproduces that integer
arithmetic, which is why `images_ori` and `images_crop` come back byte-identical rather than
within a LSB or two. A float implementation of the same filter is not good enough: the error
propagates through the vision tower and flips greedy ties on dense pages.

The SAM positional table is the other one that has to be exact. It is resampled with ATen's
antialias bicubic at `a = -0.5` (measured against torch's own `get_abs_pos_sam`: `a = -0.5` gives
max|d| 4.0e-07, `a = -0.75` gives 7.1e-03). Only the 640 tiles go through it - the 1024 global
view is the native 64x64 grid and torch early-returns, so it must come back bit-identical. That
asymmetry is why a wrong resize hides: single-view pages stay exact while every tile drifts ~2%.

## Building and publishing weights

```
cd <hf snapshot parent>
python Tools/ocr/convert.py --src model_hf --out build/publish-fidelity --policy q8
python Tools/ocr/convert.py --src model_hf --out build/publish-balanced --policy dyn-k
python Tools/ocr/convert.py --src model_hf --out build/publish-compact  --policy dyn-j
Tools/ocr/publish.sh build            # gh release upload, tag ocr-weights-v1
```

Output is sharded under 1.9 GB per file (a GitHub release asset is capped at 2 GiB) and carries
`tokenizer.json`, so an installed variant is self-contained. The app fetches `omni-ocr.json`
first and learns the shard count from it, so a re-sharded build needs no app change.

Installed variants live in `~/Library/Application Support/Omni/ocr/<variant>/`. Nothing downloads
them automatically - Settings > Storage > OCR model is the only path, and Remove is the only way
they are deleted.

## Reproducing the gate

```
# reference dumps (run from the HF snapshot's parent, needs a torch env)
python Tools/ocr/ref_dump.py --image bench/doc_small.png --out ref/doc_small
python Tools/ocr/ref_dump.py --image bench/doc_small.png --out ref-fp32/doc_small --dtype float32

# the gate itself
swift build -c release --product ocr-verify
.build/release/ocr-verify <modelDir> ref --horizon ref-fp32
```

Non-zero exit on a prompt-id mismatch, or on a text divergence EARLIER than the page's horizon.
A divergence beyond the horizon is reported and does not fail: there is nothing there to be
exact against.
