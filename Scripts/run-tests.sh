#!/bin/bash
# Build and run the OmniKit test bundle.
#
#   ./Scripts/run-tests.sh [OmniKitTests.SomeTestClass]   # filter optional
#
# Why this exists instead of a plain `xcodebuild test`:
#   1. swift-tokenizers ships its Rust FFI as an SE-0482 staticLibrary artifactbundle. xcodebuild
#      does not expose that module to the package's TokenizersFFI target, so the build fails with
#      "Cannot find type 'RustBuffer'". The fix is the same GLOBAL module-map + static-lib overrides
#      build-app.sh uses (they apply to every target, package targets included).
#   2. The SPM test bundle built that way is not code-signed, and `xcodebuild test`'s runner refuses
#      to load an unsigned bundle ("Failed to create a bundle instance representing ...xctest").
#      So we build-for-testing, ad-hoc sign the bundle, and run it directly with `xcrun xctest`.
set -euo pipefail
cd "$(dirname "$0")/.."

DD=".build/xcode-rel"   # reuse the app's derived data so the Metal toolchain / packages are warm
MODEL="${OMNI_MODEL_DIR:-/private/tmp/omni-model}"
ART="$PWD/$DD/SourcePackages/artifacts/swift-tokenizers/TokenizersRust/TokenizersRust.artifactbundle"

# ONE BUILD SCRIPT AT A TIME. This one moves Omni.xcodeproj out of the tree (below), so a
# build-app.sh or ui-test.sh running concurrently finds no project, regenerates one from
# project.yml - without OMNI_TEAM_ID, because the value it recovers lives in the project that is
# currently in /tmp - and then this script's restore moves the backup INSIDE the regenerated
# directory. The result is an unsignable project and a nested Omni.xcodeproj/Omni.xcodeproj.bak.NNN,
# from a pair of commands that each look perfectly safe on their own.
LOCK=.build/omni-build.lock
mkdir -p .build
if ! mkdir "$LOCK" 2>/dev/null; then
  holder=$(cat "$LOCK/pid" 2>/dev/null || echo "?")
  if [ "$holder" != "?" ] && ! kill -0 "$holder" 2>/dev/null; then
    echo "clearing a build lock left by dead pid $holder"
    rm -rf "$LOCK"; mkdir "$LOCK"
  else
    echo "another build script is running (pid $holder); waiting for it..."
    while ! mkdir "$LOCK" 2>/dev/null; do sleep 2; done
  fi
fi
echo $$ > "$LOCK/pid"

# The generated Omni.xcodeproj shadows the SwiftPM package for xcodebuild; move it aside and restore
# it no matter how we exit.
moved=0
if [ -d Omni.xcodeproj ]; then mv Omni.xcodeproj "/tmp/Omni.xcodeproj.bak.$$"; moved=1; fi
restore() {
  # rm -rf FIRST. `mv src dst` where dst is an existing directory moves src INSIDE it, so a
  # project that reappeared while we held ours aside would swallow the backup rather than be
  # replaced by it. The lock above should make that impossible; this makes it non-destructive
  # even if it is not.
  if [ "$moved" = 1 ]; then rm -rf Omni.xcodeproj; mv "/tmp/Omni.xcodeproj.bak.$$" Omni.xcodeproj; fi
  rm -rf "$LOCK"
}
trap restore EXIT

# Resolve packages first so the Rust artifact exists before compile.
xcodebuild -resolvePackageDependencies -scheme Omni-Package -derivedDataPath "$DD" >/dev/null

xcodebuild build-for-testing -scheme Omni-Package -destination 'platform=macOS' -derivedDataPath "$DD" \
  OTHER_SWIFT_FLAGS="\$(inherited) -Xcc -fmodule-map-file=$ART/include/module.modulemap -Xcc -I$ART/include" \
  OTHER_LDFLAGS="\$(inherited) $ART/apple-macos/libtokenizers_rust.a"

PROD="$DD/Build/Products/Debug"
BUNDLE="$PROD/OmniKitTests.xctest"
[ -d "$BUNDLE" ] || { echo "test bundle not found: $BUNDLE"; exit 1; }

# Ad-hoc sign so xctest will load it.
codesign --force --deep --sign - "$BUNDLE" >/dev/null

# Optional class/method filter passed through as -XCTest (e.g. OmniKitTests.VectorStoreTests).
if [ "$#" -gt 0 ]; then
  OMNI_MODEL_DIR="$MODEL" DYLD_FRAMEWORK_PATH="$PROD" DYLD_LIBRARY_PATH="$PROD" \
    xcrun xctest -XCTest "$1" "$BUNDLE"
else
  OMNI_MODEL_DIR="$MODEL" DYLD_FRAMEWORK_PATH="$PROD" DYLD_LIBRARY_PATH="$PROD" \
    xcrun xctest "$BUNDLE"
fi
