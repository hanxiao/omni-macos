#!/bin/zsh
# Copy the GPU kernels the APP build compiled to everywhere the SwiftPM-built tools and tests load
# them from. `swift build` does not compile Metal, so omni-verify, ocr-verify and the test bundle
# read a COPY - which nothing refreshed: on 2026-10-02 the test bundle's was from June and the
# tools' from September. Harmless while MLX stood still; wrong the moment it moved. Run after
# Scripts/build-app.sh whenever mlx-swift changes.
set -euo pipefail
cd "$(dirname "$0")/.."
SRC=.build/xcode-rel/Build/Products/Release/Omni.app/Contents/Resources/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib
[ -f "$SRC" ] || { echo "no app build at $SRC: run Scripts/build-app.sh Release first" >&2; exit 1; }
for d in .build/arm64-apple-macosx/release .build/arm64-apple-macosx/debug; do
  mkdir -p "$d"; cp "$SRC" "$d/mlx.metallib"
  [ -d "$d/mlx-swift_Cmlx.bundle/Contents/Resources" ] && cp "$SRC" "$d/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib"
  for t in "$d"/OmniPackageTests.xctest/Contents/MacOS; do
    [ -d "$t" ] || continue
    cp "$SRC" "$t/mlx.metallib"; cp "$SRC" "$t/default.metallib"
    [ -d "$t/mlx-swift_Cmlx.bundle/Contents/Resources" ] && cp "$SRC" "$t/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib"
  done
done
echo "metallib synced from $SRC ($(stat -f %z "$SRC") bytes)"
