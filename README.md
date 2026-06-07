<p align="center">
  <img src="site/omni/assets/omni-mascot.png" alt="Omni" width="180">
</p>

<h1 align="center">Omni</h1>

<p align="center">Semantic search over your local files, running entirely on-device.</p>

<p align="center">
  <a href="https://hanxiao.io/omni"><b>Download for macOS &rarr;</b></a>
</p>

Omni indexes your files and lets you search them by meaning instead of filename. A
text query finds matching documents, code, PDFs, images, audio, and video together,
because everything is embedded into one shared vector space. The model runs in-process
on Apple GPUs via a native MLX-Swift port of `jina-embeddings-v5-omni`, in two sizes -
[Nano](https://huggingface.co/jinaai/jina-embeddings-v5-omni-nano-mlx) (~1.9 GB) and
[Small](https://huggingface.co/jinaai/jina-embeddings-v5-omni-small-mlx) (~3.1 GB). No
Python, no server, no network at query time.

## Install

Download the latest DMG from [**hanxiao.io/omni**](https://hanxiao.io/omni) (or from
[GitHub Releases](https://github.com/hanxiao/omni-macos/releases)), open it, and drag
**Omni** onto **Applications**. Builds are notarized, so they open without a Gatekeeper prompt.

On first launch Omni downloads the model on-device (Nano ~1.9 GB or Small ~3.1 GB),
then runs offline. Point it at folders to index (Documents, Downloads, Desktop, or any
folder you pick), press Index, then search.

Requires an Apple silicon Mac on macOS 14 or later.

## Build from source

```
brew install xcodegen
export OMNI_TEAM_ID=XXXXXXXXXX   # your 10-char Apple Team ID (see below)
xcodegen generate
open Omni.xcodeproj              # then Cmd+R
```

You need:

- **Apple silicon Mac, macOS 14+.**
- **Xcode 26 with the Metal Toolchain** (`xcodebuild -downloadComponent MetalToolchain`).
  MLX-Swift compiles Metal shaders; a plain SwiftPM command-line build cannot, so build
  through Xcode or `xcodebuild`.
- **The model directory** (`model.safetensors`, `tokenizer.json`, `config.json`,
  `adapters/retrieval/`) from
  [`jinaai/jina-embeddings-v5-omni-small-mlx`](https://huggingface.co/jinaai/jina-embeddings-v5-omni-small-mlx)
  (or the `-nano-` variant). The app finds it via `$OMNI_MODEL_DIR`,
  `~/Library/Application Support/Omni/`, or the HuggingFace cache, and otherwise asks
  you to pick the folder.

### Why an Apple Developer account is needed

Omni reads files in your Documents, Downloads, and Desktop, which macOS gates behind
TCC permission. The app is code-signed (not ad-hoc) so the system ties that permission
to a stable signature and remembers your grant across rebuilds instead of re-prompting
every time. Signing requires a Team ID, which is why `OMNI_TEAM_ID` is set above.

- **Build and run locally:** a **free** Apple ID is enough. Add it in Xcode (Settings -
  Accounts), use the personal team it creates, and put that team's ID in `OMNI_TEAM_ID`.
- **Distribute a notarized DMG** like the Releases here: this needs the **paid Apple
  Developer Program** ($99/yr) for a *Developer ID Application* certificate and Apple's
  notary service. The release pipeline (`.github/workflows/release.yml`) uses it; you
  don't need it just to run Omni yourself.

The repository contains no Apple credentials. The Team ID comes from `OMNI_TEAM_ID`
locally and from the `APPLE_TEAM_ID` GitHub secret in CI; the signing certificate,
notary password, and deploy tokens are all GitHub Actions secrets.

## Verify the engine

The MLX-Swift encoder is checked numerically against Python reference fixtures: text
must match to cosine >= 0.999 with identical token ids; image, video, and audio towers
match the upstream `model.py` to cosine ~1.0 on identical preprocessed inputs.

```
uv run python Tools/gen_fixtures.py          # regenerate fixtures (needs mlx + tokenizers)
cp -R <model snapshot> /private/tmp/omni-model
make test                                    # compiles shaders, asserts the cosines
```

## Architecture

```
Sources/OmniKit/   engine + indexer (SPM library)
App/               SwiftUI macOS app (project.yml -> Omni.xcodeproj via XcodeGen)
Tools/             reference fixture generator
Tests/             numeric parity + end-to-end search tests
```

### Embedding

`jina-embeddings-v5-omni` ported to MLX-Swift: a Qwen3 text tower, a Qwen3-VL vision
tower (also used for video frames and scanned-PDF pages), and a Whisper-style audio
tower. `WeightStore` loads the HF safetensors, merges the retrieval LoRA into the
backbone (upcast to fp32, merge, cast back to bf16), and the encoders pool the last
token and L2-normalize. All modalities land in one shared space, with media wrapped in
a `Document:` prefix and an end-of-text suffix so cross-modal vectors align. MLX calls
are serialized through a priority gate: an interactive query jumps ahead of in-flight
indexing work, so search stays responsive while indexing runs.

### Indexing

A crawl -> extract -> chunk -> embed -> store pipeline, incremental via file mtime and
size so re-indexing only touches what changed. A concurrent decode stage (text
extraction, image patchify, audio mel) feeds a single serialized GPU embed stage.
Throughput comes from batching the GPU forward: text is chunked (~1800 chars, 200
overlap) and embedded in cross-file batches; images run one block-diagonal vision
forward per batch; audio batches clips under a frame budget; video samples a few frames
into one temporal embedding. Batches double-buffer - the next batch's GPU forward
overlaps the previous batch's host readout.

### Storing

Embeddings are stored as **bf16** (2 bytes per dimension): half the size of fp32 on
disk and in memory, with negligible recall loss on L2-normalized vectors. SQLite holds
file metadata and the persisted vectors; the live search index is a single contiguous
bf16 matrix kept in sync on every insert, update, and delete.

### Search

Exact brute-force cosine: one MLX matmul of the query against the resident bf16 matrix
on the GPU - no approximate index, no recall tradeoff. The matrix is split into a
GPU-resident **base** prefix plus a small **delta** of rows added since the base was
built, scored by a second matmul fused into one evaluation, so an indexing insert never
forces the whole matrix to be recopied per query. The base is rebuilt only on a
structural change (delete or reload) or once the delta grows past a threshold. Top-K
comes from a bounded min-heap over per-file winners rather than a full sort, and the
result is filtered by kind, folder, extension, and recency. Idle search is a few
milliseconds.

## License

[Apache 2.0](LICENSE). The model weights are covered by the upstream Jina license
(CC-BY-NC-4.0), not this repository.
