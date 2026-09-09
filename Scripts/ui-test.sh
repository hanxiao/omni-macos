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

DD=".build/xcode-rel"
ART="$PWD/$DD/SourcePackages/artifacts/swift-tokenizers/TokenizersRust/TokenizersRust.artifactbundle"

if command -v xcodegen >/dev/null 2>&1; then
  if [ ! -f Omni.xcodeproj/project.pbxproj ] || [ project.yml -nt Omni.xcodeproj/project.pbxproj ]; then
    if [ -z "${OMNI_TEAM_ID:-}" ] && [ -f Omni.xcodeproj/project.pbxproj ]; then
      OMNI_TEAM_ID=$(grep -m1 'DEVELOPMENT_TEAM = ' Omni.xcodeproj/project.pbxproj | sed -E 's/.*= ([A-Z0-9]*);/\1/')
      export OMNI_TEAM_ID
    fi
    xcodegen generate
  fi
fi

xcodebuild -resolvePackageDependencies -project Omni.xcodeproj -scheme OmniUITests \
  -derivedDataPath "$DD" >/dev/null

# Informational only - the tests decide for themselves by asking the app, because the sandboxed
# runner cannot see the user's Application Support.
OCR_MODEL=0
for d in "$HOME/Library/Application Support/Omni/ocr"/*; do
  [ -e "$d/omni-ocr.json" ] && OCR_MODEL=1
done
[ "$OCR_MODEL" = 1 ] || echo "note: OCR model not installed - the OCR UI tests will skip"

ONLY=()
if [ $# -gt 0 ]; then ONLY=(-only-testing:"OmniUITests/$1"); fi

xcodebuild -project Omni.xcodeproj -scheme OmniUITests -configuration Debug \
  -destination 'platform=macOS' -derivedDataPath "$DD" \
  "${ONLY[@]}" \
  OTHER_SWIFT_FLAGS="\$(inherited) -Xcc -fmodule-map-file=$ART/include/module.modulemap -Xcc -I$ART/include" \
  OTHER_LDFLAGS="\$(inherited) $ART/apple-macos/libtokenizers_rust.a" \
  test
