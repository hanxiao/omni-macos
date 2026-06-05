#!/bin/bash
# Build Omni.app (Release by default: ./Scripts/build-app.sh [Debug|Release]).
#
# Why this script instead of a plain `xcodebuild`:
# swift-tokenizers ships its Rust backend as an SE-0482 `staticLibrary` artifactbundle. SwiftPM
# honors the artifact's clang module map + headers, but xcodebuild does NOT expose that module,
# so `canImport(TokenizersRust)` is false in TokenizersFFI and the FFI calls go undefined
# ("cannot find 'uniffi_...' in scope"). Project-level build settings can't fix it because the
# failing target is the *package's* TokenizersFFI, which doesn't inherit our project settings.
# The fix is to pass the module map + static lib as GLOBAL command-line overrides, which apply
# to every target in the build, including package targets.
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:-Release}"
DD=".build/xcode-rel"
ART="$PWD/$DD/SourcePackages/artifacts/swift-tokenizers/TokenizersRust/TokenizersRust.artifactbundle"

# Resolve packages first so the artifact (module map + .a) exists before compile.
xcodebuild -resolvePackageDependencies -project Omni.xcodeproj -scheme Omni -derivedDataPath "$DD" >/dev/null

xcodebuild -project Omni.xcodeproj -scheme Omni -configuration "$CONFIG" \
  -destination 'platform=macOS' -derivedDataPath "$DD" \
  OTHER_SWIFT_FLAGS="\$(inherited) -Xcc -fmodule-map-file=$ART/include/module.modulemap -Xcc -I$ART/include" \
  OTHER_LDFLAGS="\$(inherited) $ART/apple-macos/libtokenizers_rust.a" \
  build

echo "Built: $PWD/$DD/Build/Products/$CONFIG/Omni.app"
