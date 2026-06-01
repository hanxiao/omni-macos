# Omni

Native macOS app for semantic search over your local files. Embeddings run
in-process via a native MLX-Swift port of
[`jinaai/jina-embeddings-v5-omni-small-mlx`](https://huggingface.co/jinaai/jina-embeddings-v5-omni-small-mlx).
No Python, no server, no network at query time.

- Qwen3 text tower ported to MLX-Swift, retrieval LoRA merged at load time.
- Last-token pooling, L2 normalization, Matryoshka dimensions (1024 default).
- Numerically identical to the Python reference: cosine 1.00000 on the fixture set.
- Qwen3-VL vision tower also ported: scanned PDFs (no text layer) and image files
  route to the vision path and land in the same embedding space as text. Tower
  parity vs the Python `encode_image` reference is cosine 1.00000; full
  CGImage-to-embedding is cosine 0.985 (the gap is CoreGraphics vs PIL resize).
- SQLite vector store with brute-force cosine search.
- Indexes plain text, source code, Markdown, PDFs, office documents, and images.

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
