#!/bin/bash
# Publish OCR weight variants as GitHub release assets.
#
# The app fetches from https://github.com/<repo>/releases/download/<tag>/<variant>-<file>, so the
# asset names encode the variant: release assets are a flat namespace with no directories.
#
# `gh release upload file#name` sets the asset's LABEL, not its name - the asset keeps the file's
# basename, and every URL the app builds 404s. The name is set afterwards through the API, which
# renames in place rather than re-uploading gigabytes.
#
# A single asset is capped at 2 GiB, which is why the converter shards under 1.9 GB. Nothing here
# re-shards; it uploads what convert.py produced.
#
#   Tools/ocr/publish.sh <buildRoot> [variant ...]
#
# where <buildRoot> holds publish-<variant>/ directories. With no variants named, publishes all
# three. This uploads several GB per variant - run it deliberately.
set -euo pipefail

REPO="${OCR_REPO:-hanxiao/omni-macos}"
TAG="${OCR_TAG:-ocr-weights-v1}"
ROOT="${1:?usage: publish.sh <buildRoot> [variant ...]}"
shift || true
VARIANTS=("$@")
if [ ${#VARIANTS[@]} -eq 0 ]; then VARIANTS=(fidelity balanced compact); fi

if ! gh release view "$TAG" --repo "$REPO" >/dev/null 2>&1; then
  echo "creating release $TAG"
  gh release create "$TAG" --repo "$REPO" \
    --title "jina-ocr-v1 MLX weights" \
    --notes "MLX-Swift weights for the optional OCR model in Omni. Converted from jinaai/jina-ocr-v1 by Tools/ocr/convert.py; see docs/OCR.md for the quantization policies and the measured fidelity of each variant." \
    --latest=false
fi

for variant in "${VARIANTS[@]}"; do
  dir="$ROOT/publish-$variant"
  [ -d "$dir" ] || { echo "missing $dir"; exit 1; }
  echo "== $variant ($(du -sh "$dir" | cut -f1))"
  # omni-ocr.json is what the downloader fetches FIRST to learn the shard count, so it has to be
  # present for any of the rest to be reachable. Upload it last so a half-finished upload never
  # advertises shards that are not there yet.
  for f in "$dir"/*.safetensors "$dir"/tokenizer.json "$dir"/tokenizer_config.json "$dir"/omni-ocr.json; do
    [ -f "$f" ] || continue
    base="$(basename "$f")"
    name="jina-ocr-v1-mlx-$variant-$base"
    echo "   $name"
    gh release upload "$TAG" "$f" --repo "$REPO" --clobber
    id=$(gh api "/repos/$REPO/releases/tags/$TAG" -q ".assets[] | select(.name == \"$base\") | .id")
    [ -n "$id" ] && gh api -X PATCH "/repos/$REPO/releases/assets/$id" -f name="$name" -q '.name' >/dev/null
  done
done

echo "done: https://github.com/$REPO/releases/tag/$TAG"
