#!/bin/bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
MODE=${1:-run}
RUNTIME="$HOME/Library/Application Support/OmniEmbeddingGemma2/runtime"
if ! "$RUNTIME/venv/bin/python" -c 'from transformers import EmbeddingGemma2Processor; from mlx_vlm.embedding_loader import load_embedding_model' >/dev/null 2>&1; then
  ./Scripts/setup-embeddinggemma2.sh
fi
cp Runtime/EmbeddingGemma2/worker.py "$RUNTIME/worker.py"
./Scripts/build-app.sh Debug
BUNDLE="$(cd "$ROOT/.build/xcode-rel/Build/Products/Debug" && pwd -P)/Omni.app"
# Stop only this fork; do not terminate an upstream Omni instance.
pkill -f "^$BUNDLE/Contents/MacOS/Omni" >/dev/null 2>&1 || true
case "$MODE" in
  run) /usr/bin/open -n "$BUNDLE" ;;
  --verify) /usr/bin/open -n "$BUNDLE"; sleep 2; pgrep -f "$BUNDLE/Contents/MacOS/Omni" >/dev/null ;;
  --debug) lldb -- "$BUNDLE/Contents/MacOS/Omni" ;;
  --logs) /usr/bin/open -n "$BUNDLE"; /usr/bin/log stream --info --style compact --predicate 'process == "Omni"' ;;
  --telemetry) /usr/bin/open -n "$BUNDLE"; /usr/bin/log stream --info --style compact --predicate 'subsystem == "io.hanxiao.omni"' ;;
  *) echo "usage: $0 [run|--verify|--debug|--logs|--telemetry]" >&2; exit 2 ;;
esac
