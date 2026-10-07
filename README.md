<p align="center">
  <img src="site/omni/assets/omni-mascot.png" alt="Omni" width="180">
</p>

<h1 align="center">Omni</h1>

<p align="center">Search every file on your Mac by meaning, with local multimodal embeddings.</p>

This fork of [hanxiao/omni-macos](https://github.com/hanxiao/omni-macos) uses
**Google EmbeddingGemma 2** as its default search model. Text (including French),
images, audio and video share normalized 768-dimensional vectors. Jina Small/Nano
remain explicit alternatives in Settings.

The app remains Swift/SwiftUI. EmbeddingGemma 2 runs on Apple Silicon through a
persistent **local Python/MLX helper**, connected by private pipes. Inference works
offline; the helper exposes no HTTP endpoint and does not upload inputs. This is
not yet a self-contained downloadable app: install the helper before launching.
Upstream releases and published Jina benchmarks do not describe this fork.

## Install and run this fork

Requirements: Apple Silicon, macOS 14+, Xcode with the Metal Toolchain,
XcodeGen, and Python 3.12 (or `uv`, which manages it).

```bash
brew install xcodegen uv
./Scripts/setup-embeddinggemma2.sh
export OMNI_TEAM_ID=XXXXXXXXXX  # Apple development team, local signing
./script/build_and_run.sh
```

On first launch the app downloads the official Google checkpoint (~1.5 GB plus
its tokenizer/processor). Model and helper downloads require an internet
connection; subsequent inference uses only local files. An existing local copy
can be selected with `OMNI_MODEL_DIR`.

The fork uses a separate bundle identifier, preferences and default data folder
(`~/Library/Application Support/OmniEmbeddingGemma2`). Index filenames include
the model variant; old Jina vectors are never reused as Google vectors.
Automatic image/video tags remain a Jina-only feature. Search using images,
video and audio is supported by EmbeddingGemma 2.

See [runtime details and validation](docs/embeddinggemma2.md) for pinned versions,
tests, resource behavior and current limitations.

## Serving

Settings > Serving exposes an HTTP API on `127.0.0.1:51234` with an MCP endpoint at `/mcp`.

```bash
curl -sX POST localhost:51234/v1/search -d '{"query": "invoice from march"}'
curl -sX POST localhost:51234/v1/sources/add -d '{"path": "~/Projects"}'
```

## Build

```bash
brew install xcodegen
export OMNI_TEAM_ID=XXXXXXXXXX   # your Apple Team ID; a free account works for local builds
./Scripts/build-app.sh           # not a bare xcodebuild: see the script's header
```

Needs Xcode 26 with the Metal Toolchain. The app downloads its models itself; `OMNI_MODEL_DIR`
points it at a local copy instead. `make test` runs the suite, including numeric parity with Python reference
fixtures.

## Citation

```bibtex
@inproceedings{xiao2026omnimacos,
  title     = {omni-macos: On-Device Omni-Modal Search on Apple Silicon},
  author    = {Xiao, Han},
  booktitle = {NeurIPS 2026 Workshop on On-Device Intelligence},
  year      = {2026},
  eprint    = {2608.05543},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2608.05543}
}
```

## License

[Apache 2.0](LICENSE). The default Google EmbeddingGemma 2 weights are Apache 2.0.
Optional Jina weights retain their upstream license (CC-BY-NC-4.0).
