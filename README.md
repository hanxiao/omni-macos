# Omni

Native macOS app for semantic search over your local files. Embeddings run
in-process via a native MLX-Swift port of
[`jinaai/jina-embeddings-v5-omni-small-mlx`](https://huggingface.co/jinaai/jina-embeddings-v5-omni-small-mlx).
No Python, no server, no network at query time.

- Qwen3 text tower ported to MLX-Swift, retrieval LoRA merged at load time.
- Last-token pooling, L2 normalization, Matryoshka dimensions (1024 default).
- All four modalities ported to MLX-Swift and verified against the original
  v5-omni `model.py` to **cosine 1.00000** (identical preprocessed inputs):
  text, image (vision tower), video (`encode_video`, temporal), audio
  (`encode_audio`, Whisper-style tower). Image end-to-end from a CGImage is 0.985,
  the only non-1.0 number, and the gap is CoreGraphics-vs-PIL resize on the input
  pixels, not the model.
- Everything lands in one shared embedding space, so a text query finds images,
  video, audio, scanned PDFs, and documents together.
- SQLite vector store with brute-force cosine search.
- Indexes images, video, and audio by default; text/code/PDF/office is an opt-in
  toggle (per-modality switches in the sidebar).
- Finder-style results: List and Gallery views with real QuickLook thumbnails.
- Search filters by file kind, folder, extension, and minimum score.

## Install

Download the latest `Omni-*.dmg` from
[Releases](https://github.com/hanxiao/omni-macos/releases), open it, and drag
**Omni** onto **Applications**.

Omni is not notarized by Apple yet, so macOS warns the first time you launch a
downloaded copy. To open it (once): open **System Settings - Privacy & Security**,
scroll down, and click **Open Anyway** next to Omni, then confirm. Or run once in
Terminal: `xattr -dr com.apple.quarantine /Applications/Omni.app`

On first run Omni downloads the search model (Nano ~1.9 GB or Small ~3.1 GB) on-device.

## Requirements

- Apple silicon Mac, macOS 14+.
- Xcode 26 with the Metal Toolchain component
  (`xcodebuild -downloadComponent MetalToolchain`). SwiftPM command-line builds
  cannot compile the Metal shaders; use Xcode / `xcodebuild`.
- The omni model directory (model.safetensors, tokenizer.json, config.json,
  adapters/retrieval/). Get it from HuggingFace:
  `jinaai/jina-embeddings-v5-omni-small-mlx`.

## Build and run

```
brew install xcodegen
xcodegen generate
open Omni.xcodeproj      # then Cmd+R
```

On first launch Omni looks for the model in `$OMNI_MODEL_DIR`,
`~/Library/Application Support/Omni/model`, and the HuggingFace cache. If none is
found it asks you to pick the folder. Add folders to index from the sidebar,
press Index, then search.

## Verify the engine

The MLX-Swift encoder is checked against Python reference fixtures.

```
# 1. regenerate fixtures from the reference model (needs mlx + tokenizers)
uv run python Tools/gen_fixtures.py

# 2. stage the model where the sandboxed test runner can read it
cp -R <model snapshot> /private/tmp/omni-model

# 3. run the parity test (compiles the Metal shaders, asserts cosine >= 0.999)
xcodebuild test -scheme Omni-Package -destination 'platform=macOS' -only-testing:OmniKitTests
```

## Layout

```
Package.swift            OmniKit library + omni-verify + tests
Sources/OmniKit/         engine (OmniTextEncoder, WeightStore, OmniImageEncoder),
                         indexer (FileCrawler, FileExtractor, VectorStore, Indexer)
App/                     SwiftUI macOS app (project.yml -> Omni.xcodeproj)
Tools/gen_fixtures.py    reference fixture generator
Tests/OmniKitTests/      numeric parity test
```

## License

Model weights are under the upstream jina license (CC-BY-NC-4.0). Application
code in this repository is MIT.
