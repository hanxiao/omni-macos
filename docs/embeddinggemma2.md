# EmbeddingGemma 2 integration

## Runtime and reproducibility

The official [Google model card](https://huggingface.co/google/embeddinggemma-2)
describes the model and its Apache 2.0 license. The checkpoint is pinned to
`914f7f89142e33e77833254d9c9b90c3cef7303b` in `ModelDownloader`. It is loaded
in BF16 without Jina adapters or quantization. Google outputs 768-dimensional,
L2-normalized vectors. The Python runtime uses MLX 0.32.3 and pinned commits of
MLX-VLM and Transformers; see `Runtime/EmbeddingGemma2/requirements.txt`.
The existing native Swift MLX runtime remains in use for vector search and Jina.

The helper starts once per engine, reads newline-delimited JSON from stdin and
returns JSON on stdout. Images are lossless PNG bytes, audio is mono 16 kHz
float32 PCM, and video is a sampled frame sequence. Google performs its own
media preprocessing. The Jina Qwen patch and log-mel preprocessors are bypassed.
Text uses Google's retrieval prefixes (`task: search result | query: ` and
`title: none | text: `). Media uses the processor's modality tokens without
text prefixes. Text is truncated to the model's 8192-token context; Omni's
existing text chunking remains in use. Scanned-PDF queries average normalized
page vectors and renormalize the result.

Audio indexing uses Omni's existing bounded segments (up to 240 seconds); an
audio-file query uses its first segment. The released processor configuration
has a 280-token audio cap, so the helper raises it to the model's 8192-token
context. The opt-in test includes 35-second PCM to catch feature/token mismatch.

Inference sets `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE`. No server is opened.
Errors and library diagnostics go to per-worker logs under
`~/Library/Application Support/OmniEmbeddingGemma2/runtime/logs` (permissions
0600); corpus bytes are passed in memory. Invalid/nonfinite vectors are rejected.
A response timeout terminates the helper so subsequent responses cannot be
mistaken for another request. Relaunch/reload the engine after a fatal helper
failure. A missing helper gives an actionable setup error.

## Resource behavior and limitations

The helper serializes model requests and clears its MLX buffer cache at idle.
Its cache is bounded to 256 MiB; the user's memory setting is applied to the
helper's MLX allocator as well as the native allocator. These are per-process
limits, not a combined operating-system cap. The memory breakdown includes the
helper's macOS physical footprint and the last reported helper MLX counters.
Counters are updated after requests, so slices can lag a request in progress.

Jina-specific patch-tag extraction is unavailable with Google and its UI toggle
is disabled. EmbeddingGemma 2 does not produce captions/transcripts; semantic
media search operates directly on embeddings. The native Jina GPU scheduling,
in-process query graph and throughput benchmarks do not apply to the helper.
The upstream automatic update feed is disabled so an upstream Jina binary cannot
replace this fork. Manual update checks explain how to update the fork.
Python dependency setup is currently a developer installation step; the helper
and model must be packaged separately before distributing a standalone DMG.

Build scripts keep derived data in a per-checkout folder under
`~/Library/Caches/OmniEmbeddingGemma2` and link `.build/xcode-rel` to it. This
avoids synchronized Documents folders adding Finder attributes that invalidate
code signing. Existing build output is moved, not deleted.

## Verification

Run a real checkpoint integration test on an Apple Silicon Mac:

```bash
OMNI_GEMMA_INTEGRATION=1 \
OMNI_MODEL_DIR="$HOME/Library/Application Support/OmniEmbeddingGemma2/embeddinggemma2" \
./Scripts/run-tests.sh OmniKitTests.EmbeddingGemmaIntegrationTests
```

This checks four French retrieval pairs, finite unit-length text/image/audio/video
vectors, 35-second raw audio, an indexed six-file mixed corpus, and tower
selection. It is a smoke test, not a French benchmark, a Google-vs-Jina ranking,
or evidence of speech understanding accuracy. Upstream parity fixtures concern
Jina and should be run with a Jina checkpoint.

`./script/build_and_run.sh --verify` builds, launches and checks the app process.
A process check alone does not prove successful media search in the interface.

For optional numerical parity against the Transformers FP32 CPU reference:

```bash
"$HOME/Library/Application Support/OmniEmbeddingGemma2/runtime/venv/bin/python" \
  Scripts/check-embeddinggemma2-parity.py
```

The check uses small text/image/video/audio fixtures and requires cosine similarity
above 0.99. It loads both models, so allow additional memory. On the development
M4 Mac, observed cosines were 0.999971 (French text), 0.999980 (image), 0.999670
(video) and 0.999960 (audio). This checks runtime fidelity on these inputs, not a
multilingual retrieval benchmark or general media recognition accuracy.
