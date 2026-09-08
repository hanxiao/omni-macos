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
| `Sources/ocr-verify/` | the numeric gate and benchmark |
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

| build | size | exact | mean CER | decode | vs torch |
|---|---|---|---|---|---|
| torch bf16, MPS (reference) | 7.4 GB | - | - | 38.3 tok/s | 1.0x |
| bf16, no quantization | 6.67 GB | 3/7 | 0.0231 | 161.3 tok/s | 4.2x |
| **fidelity** (`q8`) | 4.55 GB | 3/7 | 0.0159 | 163.5 tok/s | 4.3x |
| **balanced** (`dyn-k`) | 4.46 GB | 3/7 | 0.0321 | 184.7 tok/s | 4.8x |
| **compact** (`dyn-j`, dynamic 4-bit) | 4.06 GB | 2/7 | 0.0469 | 186.1 tok/s | 4.9x |

TTFT on M3 Ultra: 191-233 ms single-view, 481-749 ms multi-crop (2x3 tiles, 1007 prompt tokens);
the quantized builds pay ~25% more TTFT than bf16 because the fused MoE prefill is slower on
packs, and win it back several times over in decode.

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
* **The FastMTP draft head**: 734 MB, and speculative decoding measured slower than greedy at
  every draft length. Verifying k tokens activates ~k x top_k distinct experts, and cost here
  scales with active experts, not tokens. `--mtp` includes it; the default does not.

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
