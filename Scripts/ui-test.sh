#!/bin/bash
# Run the XCUITest suites against the real app.
#
#   ./Scripts/ui-test.sh                       # every UI test
#   ./Scripts/ui-test.sh OCRWorkspaceUITests   # one class
#   ./Scripts/ui-test.sh OCRWorkspaceUITests/testTranscribesAPageAndCopiesItAsMarkdown
#
# Same reason this exists as Scripts/build-app.sh: swift-tokenizers ships its Rust backend as an
# SE-0482 artifactbundle that xcodebuild does not expose as a module, so a plain `xcodebuild test`
# fails inside the PACKAGE's own target with "cannot find type 'RustBuffer' in scope". The module
# map and static library have to be passed as global command-line overrides, which reach package
# targets too.
#
# These drive the real app, so macOS shows its automation banner and the run takes over the
# pointer. The OCR tests skip themselves when the optional model is not installed.
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

# Overridable so a second, CONCURRENT xcodebuild (the automation-mode holder) does not fight this
# one for the derived-data lock.
DD=${OMNI_DD:-.build/xcode-rel}
ART="$PWD/$DD/SourcePackages/artifacts/swift-tokenizers/TokenizersRust/TokenizersRust.artifactbundle"

if command -v xcodegen >/dev/null 2>&1; then
  if [ ! -f Omni.xcodeproj/project.pbxproj ] || [ project.yml -nt Omni.xcodeproj/project.pbxproj ]; then
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
    xcodegen generate
  fi
fi

xcodebuild -resolvePackageDependencies -project Omni.xcodeproj -scheme OmniUITests \
  -derivedDataPath "$DD" >/dev/null

# Informational only - the tests decide for themselves by asking the app, because the sandboxed
# runner cannot see the user's Application Support.
OCR_MODEL=0
# The weights live beside the embedding model as Omni/jina-ocr-v1-<slug>, not in an ocr/ of their
# own - migrateLegacyInstall moved them at launch long ago. This still probed the legacy path, so
# it printed "not installed" on every machine that has the model.
for d in "$HOME/Library/Application Support/Omni"/jina-ocr-v1-*; do
  [ -e "$d/omni-ocr.json" ] && OCR_MODEL=1
done
[ "$OCR_MODEL" = 1 ] || echo "note: OCR model not installed - the OCR UI tests will skip"

# `${ONLY[@]}` on an EMPTY array is an unbound variable under `set -u` in bash 3.2, which is
# what /bin/bash is on macOS. The script died on that line before xcodebuild ever ran - and it
# died with exit 0, so a no-arg run looked exactly like a passing test run. `${ONLY[@]+...}`
# expands to nothing when the array is empty instead of erroring.
ONLY=()
if [ $# -gt 0 ]; then ONLY=(-only-testing:"OmniUITests/$1"); fi

xcodebuild -project Omni.xcodeproj -scheme OmniUITests -configuration Debug \
  -destination 'platform=macOS' -derivedDataPath "$DD" \
  ${ONLY[@]+"${ONLY[@]}"} \
  OTHER_SWIFT_FLAGS="\$(inherited) -Xcc -fmodule-map-file=$ART/include/module.modulemap -Xcc -I$ART/include" \
  OTHER_LDFLAGS="\$(inherited) $ART/apple-macos/libtokenizers_rust.a" \
  test
