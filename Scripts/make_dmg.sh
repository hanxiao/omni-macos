#!/bin/bash
# Package a built Omni.app into a drag-to-install DMG.
#   ./Scripts/make_dmg.sh [path/to/Omni.app] [version]
# Defaults: the Release build under .build, and the app's CFBundleShortVersionString.
#
# The DMG mounts to a volume containing Omni.app plus an /Applications symlink, so the
# user drags the app onto Applications. No extra tooling required (uses hdiutil).
set -euo pipefail
cd "$(dirname "$0")/.."

APP="${1:-.build/xcode-rel/Build/Products/Release/Omni.app}"
[ -d "$APP" ] || { echo "app not found: $APP (build it first with ./Scripts/build-app.sh Release)"; exit 1; }

VERSION="${2:-$(/usr/libexec/PlistBuddy -c 'Print CFBundleShortVersionString' "$APP/Contents/Info.plist")}"
NAME="Omni"
OUT="dist"
DMG="$OUT/${NAME}-${VERSION}.dmg"

mkdir -p "$OUT"
rm -f "$DMG"

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
cp -R "$APP" "$STAGE/$NAME.app"
ln -s /Applications "$STAGE/Applications"

# First-launch help: the app is not notarized yet, so macOS warns once on a downloaded copy.
cat > "$STAGE/How to Open Omni.txt" <<'TXT'
First launch on macOS

Omni is not yet notarized by Apple, so macOS shows a warning the first time
you open a downloaded copy. To open it (you only do this once):

  1. Drag Omni onto Applications.
  2. Open System Settings - Privacy & Security.
  3. Scroll down, click "Open Anyway" next to Omni, then confirm.

Or run this once in Terminal:

  xattr -dr com.apple.quarantine /Applications/Omni.app

After that Omni opens normally. On first run it downloads the search model.
TXT

# Compressed read-only DMG (UDZO) named "Omni <version>".
hdiutil create -volname "$NAME $VERSION" -srcfolder "$STAGE" -fs HFS+ -format UDZO -ov "$DMG" >/dev/null

echo "Created $DMG ($(du -h "$DMG" | cut -f1))"
