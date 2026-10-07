#!/bin/bash
# Called while holding .build/omni-build.lock. Keep build outputs outside synced Documents.
set -euo pipefail
mkdir -p .build
if [ ! -L .build/xcode-rel ]; then
  REPO_KEY=$(printf '%s' "$PWD" | shasum -a 256 | cut -c1-16)
  CACHE="$HOME/Library/Caches/OmniEmbeddingGemma2/$REPO_KEY/xcode-rel"
  mkdir -p "$(dirname "$CACHE")"
  if [ -d .build/xcode-rel ]; then
    if [ -e "$CACHE" ]; then
      echo "Both .build/xcode-rel and $CACHE exist; preserve both and choose a single build cache." >&2
      exit 1
    fi
    mv .build/xcode-rel "$CACHE"
  else
    mkdir -p "$CACHE"
  fi
  ln -s "$CACHE" .build/xcode-rel
fi
