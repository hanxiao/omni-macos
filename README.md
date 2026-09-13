<p align="center">
  <img src="site/omni/assets/omni-mascot.png" alt="Omni" width="180">
</p>

<h1 align="center">Omni</h1>

<p align="center">Semantic search over your local files, running entirely on-device.</p>

<p align="center">
  <a href="https://arxiv.org/abs/2608.05543"><b>Technical report</b></a>
  &nbsp;&nbsp;&middot;&nbsp;&nbsp;
  <a href="https://hanxiao.io/omni"><b>Download for macOS &rarr;</b></a>
</p>

Omni indexes your files and lets you search them by meaning instead of filename. A
text query finds matching documents, code, PDFs, images, audio, and video together,
because everything is embedded into one shared vector space. The model runs in-process
on Apple GPUs via a native MLX-Swift port of `jina-embeddings-v5-omni`, in two sizes -
[Nano](https://huggingface.co/jinaai/jina-embeddings-v5-omni-nano-mlx) (~1.9 GB) and
[Small](https://huggingface.co/jinaai/jina-embeddings-v5-omni-small-mlx) (~3.1 GB). No
Python, no server, no cloud: the model downloads once, then indexing and search run with
no network at all. Airgap the Mac and Omni keeps working.

<p align="center">
  <a href="https://hanxiao.io/omni/assets/omni-intro.mp4" title="Watch the Omni demo (37 seconds)">
    <img src="site/omni/assets/omni-poster-play.jpg" alt="Watch the Omni demo: search by meaning, any file to any file, deep PDF search, folder maps, and the on-device MLX architecture" width="720">
  </a>
  <br>
  <a href="https://hanxiao.io/omni/assets/omni-intro.mp4"><b>&#9654;&#65039; Watch the 37-second demo</b></a>
</p>

## Install

Download the latest DMG from [**hanxiao.io/omni**](https://hanxiao.io/omni) (or from
[GitHub Releases](https://github.com/hanxiao/omni-macos/releases)), open it, and drag
**Omni** onto **Applications**. Builds are notarized, so they open without a Gatekeeper prompt.

On first launch Omni downloads the model once (Nano ~1.9 GB or Small ~3.1 GB). That is the
only time it touches the network: after that, both indexing and search run on-device with
nothing leaving your Mac, so you can pull the plug and run it fully airgapped. Point it at
folders to index (Documents, Downloads, Desktop, or any folder you pick), press Index, then search.

Your Apple Photos library can be indexed directly, without exporting anything: **Add photos** in
the sidebar, then choose the whole library or particular albums. Omni reads it through PhotoKit,
so edits, albums and filenames are the ones Photos shows - and iCloud photos that are not
downloaded to this Mac are skipped rather than pulled down (the same policy as Settings > iCloud
for files).

Requires an Apple silicon Mac on macOS 14 or later.

## Architecture

```
Sources/OmniKit/   engine + indexer (SPM library)
App/               SwiftUI macOS app (project.yml -> Omni.xcodeproj via XcodeGen)
Tools/             reference fixture generator
Tests/             numeric parity + end-to-end search tests
```

How the engine, indexer, store and search work is in the
[technical report](https://arxiv.org/abs/2608.05543).

## Serving: search and manage the index from anything

Omni serves an HTTP API on `127.0.0.1:51234` (Settings > Serving; LAN scope and a bearer token are
optional), including an MCP endpoint at `/mcp` so an agent can use it as a tool server.

Searching is `POST /v1/search`, plus OpenAI-, Jina- and Gemini-shaped embedding routes. Managing
what gets indexed is `/v1/sources`, which is the sidebar's four actions over HTTP - the API calls
the same code the buttons do, so an API-added folder is canonicalized, persisted, watched and
queued exactly like a dropped one.

```bash
# What is indexed, what is still indexing, and which photo albums could be added
curl -s localhost:51234/v1/sources

# Index a folder, the whole Apple Photos library, or one album
curl -sX POST localhost:51234/v1/sources/add  -d '{"path": "~/Projects"}'
curl -sX POST localhost:51234/v1/sources/add  -d '{"album": "all"}'

# Pause one without losing what it already indexed (paused defaults to true), then resume
curl -sX POST localhost:51234/v1/sources/pause -d '{"key": "/Users/me/Projects"}'
curl -sX POST localhost:51234/v1/sources/pause -d '{"key": "/Users/me/Projects", "paused": false}'

# Stop indexing it and drop its rows. Files on disk are never touched.
curl -sX POST localhost:51234/v1/sources/remove -d '{"key": "/Users/me/Projects"}'
```

Every mutation answers with the full new source list, because the caller's next question is always
"so what is indexed now" - and because a key is canonicalized on the way in, so echoing the request
back would often be a lie. The same four operations are MCP tools (`list_sources`, `add_source`,
`pause_source`, `remove_source`) alongside `search`, `search_inline`, `file_status` and `tag_image`.

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

### Verify the engine

The MLX-Swift encoder is checked numerically against Python reference fixtures: text
must match to cosine >= 0.999 with identical token ids; image, video, and audio towers
match the upstream `model.py` to cosine ~1.0 on identical preprocessed inputs.

```
uv run python Tools/gen_fixtures.py          # regenerate fixtures (needs mlx + tokenizers)
cp -R <model snapshot> /private/tmp/omni-model
make test                                    # compiles shaders, asserts the cosines
```

## Citation

If you use this software in your research, please cite:

```bibtex
@article{xiao2026omnimacos,
  title   = {omni-macos: On-Device Omni-Modal Search on Apple Silicon},
  author  = {Xiao, Han},
  journal = {arXiv preprint arXiv:2608.05543},
  year    = {2026},
  url     = {https://arxiv.org/abs/2608.05543}
}
```

## License

[Apache 2.0](LICENSE). The model weights are covered by the upstream Jina license
(CC-BY-NC-4.0), not this repository.
