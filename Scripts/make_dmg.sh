#!/bin/bash
# Package a built Omni.app into a drag-to-install DMG with the classic installer layout
# (app icon + arrow -> Applications, on a background image).
#   ./Scripts/make_dmg.sh [path/to/Omni.app] [version]
# Defaults: the Release build under .build, and the app's CFBundleShortVersionString.
#
# Uses dmgbuild (via uv), which writes the .DS_Store directly - no Finder/AppleScript automation, so
# it works headless on the self-hosted CI runner. Set NOTARIZED=1 to skip the first-launch help note
# (notarized builds open with no Gatekeeper warning).
set -euo pipefail
cd "$(dirname "$0")/.."

APP="${1:-.build/xcode-rel/Build/Products/Release/Omni.app}"
[ -d "$APP" ] || { echo "app not found: $APP (build it first with ./Scripts/build-app.sh Release)"; exit 1; }

VERSION="${2:-$(/usr/libexec/PlistBuddy -c 'Print CFBundleShortVersionString' "$APP/Contents/Info.plist")}"
NAME="Omni"
OUT="dist"
DMG="$OUT/${NAME}-${VERSION}.dmg"
BG="$PWD/Scripts/dmg/background.png"

mkdir -p "$OUT"
rm -f "$DMG"

EXTRA_ARGS=()
TMP_EXTRA=""
if [ "${NOTARIZED:-0}" != "1" ]; then
  TMP_EXTRA="$(mktemp -d)/How to Open Omni.txt"
  cat > "$TMP_EXTRA" <<'TXT'
First launch on macOS

This build is not notarized, so macOS warns the first time you open a downloaded
copy. To open it (once):

  1. Drag Omni onto Applications.
  2. Open System Settings - Privacy & Security, scroll down, click "Open Anyway".

Or run once in Terminal:

  xattr -dr com.apple.quarantine /Applications/Omni.app

On first run Omni downloads the search model.
TXT
  EXTRA_ARGS=(-D "extra=$TMP_EXTRA")
fi
trap '[ -n "$TMP_EXTRA" ] && rm -rf "$(dirname "$TMP_EXTRA")" || true' EXIT

uv run --with dmgbuild dmgbuild \
  -s Scripts/dmg/settings.py \
  -D "app=$APP" -D "bg=$BG" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
  "$NAME $VERSION" "$DMG" >/dev/null

echo "Created $DMG ($(du -h "$DMG" | cut -f1))"
