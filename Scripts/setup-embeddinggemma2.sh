#!/bin/bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
RUNTIME="$HOME/Library/Application Support/OmniEmbeddingGemma2/runtime"
mkdir -p "$RUNTIME"
if command -v uv >/dev/null; then
  uv venv --python 3.12 --allow-existing "$RUNTIME/venv"
  uv pip install --python "$RUNTIME/venv/bin/python" -r "$ROOT/Runtime/EmbeddingGemma2/requirements.txt"
else
  python3 -m venv "$RUNTIME/venv"
  "$RUNTIME/venv/bin/python" -m pip install -r "$ROOT/Runtime/EmbeddingGemma2/requirements.txt"
fi
cp "$ROOT/Runtime/EmbeddingGemma2/worker.py" "$RUNTIME/worker.py"
"$RUNTIME/venv/bin/python" -c 'from transformers import EmbeddingGemma2Processor; from mlx_vlm.embedding_loader import load_embedding_model; print("EmbeddingGemma 2 runtime ready")'
