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

# Shared with run-tests.sh, which moves Omni.xcodeproj out of the tree while it runs. Regenerating
# a project while that is happening produces one with no development team, and its restore then
# nests the backup inside ours. See the comment in run-tests.sh.
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
trap 'rm -rf "$LOCK"' EXIT

CONFIG="${1:-Release}"
DD=".build/xcode-rel"
ART="$PWD/$DD/SourcePackages/artifacts/swift-tokenizers/TokenizersRust/TokenizersRust.artifactbundle"

# The .xcodeproj is GENERATED from project.yml and gitignored. If project.yml is newer (e.g. the
# release CI bumped MARKETING_VERSION), a stale project silently stamps local builds with an old
# version string (code is current, the About/crash-report version lies). Regenerate when outdated.
if command -v xcodegen >/dev/null 2>&1; then
  if [ ! -f Omni.xcodeproj/project.pbxproj ] || [ project.yml -nt Omni.xcodeproj/project.pbxproj ]; then
    # project.yml reads OMNI_TEAM_ID at generation time; regenerating without it would drop the
    # development team and fail signing. Preserve the team from the existing project if the env
    # is not set (the public repo never contains it - this stays local-only).
    # RECOVER THE TEAM, AND REFUSE TO GENERATE WITHOUT ONE. project.yml writes
    # DEVELOPMENT_TEAM = "${OMNI_TEAM_ID}" verbatim when the variable is unset, which produces a
    # project that cannot sign - and the recovery below then reads that literal back as the team,
    # so every later run regenerates the same broken project and the only symptom is
    # "Signing for \"Omni\" requires selecting either a development team". Match the shape of a
    # real team id, and stop rather than write a project that is guaranteed not to build.
    if [ -z "${OMNI_TEAM_ID:-}" ] && [ -f Omni.xcodeproj/project.pbxproj ]; then
      OMNI_TEAM_ID=$(grep -m1 'DEVELOPMENT_TEAM = ' Omni.xcodeproj/project.pbxproj \
                     | sed -E 's/.*= "?([A-Z0-9]{10})"?;.*/\1/')
      case "$OMNI_TEAM_ID" in [A-Z0-9][A-Z0-9][A-Z0-9][A-Z0-9][A-Z0-9][A-Z0-9][A-Z0-9][A-Z0-9][A-Z0-9][A-Z0-9]) ;; *) OMNI_TEAM_ID="" ;; esac
      export OMNI_TEAM_ID
    fi
    if [ -z "${OMNI_TEAM_ID:-}" ]; then
      echo "OMNI_TEAM_ID is not set and could not be read from Omni.xcodeproj." >&2
      echo "Run: OMNI_TEAM_ID=<your 10-char team id> $0" >&2
      exit 1
    fi
    echo "project.yml newer than Omni.xcodeproj - regenerating (xcodegen, team: ${OMNI_TEAM_ID:-none})"
    xcodegen generate
  fi
fi

# Resolve packages first so the artifact (module map + .a) exists before compile.
xcodebuild -resolvePackageDependencies -project Omni.xcodeproj -scheme Omni -derivedDataPath "$DD" >/dev/null

# Any extra args after the config are passed straight to xcodebuild (e.g. CI signing overrides
# like CODE_SIGN_IDENTITY=- CODE_SIGNING_REQUIRED=NO, since CI has no "Apple Development" cert).
xcodebuild -project Omni.xcodeproj -scheme Omni -configuration "$CONFIG" \
  -destination 'platform=macOS' -derivedDataPath "$DD" \
  OTHER_SWIFT_FLAGS="\$(inherited) -Xcc -fmodule-map-file=$ART/include/module.modulemap -Xcc -I$ART/include" \
  OTHER_LDFLAGS="\$(inherited) $ART/apple-macos/libtokenizers_rust.a" \
  "${@:2}" \
  build

echo "Built: $PWD/$DD/Build/Products/$CONFIG/Omni.app"
