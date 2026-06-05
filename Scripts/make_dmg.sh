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

# Compressed read-only DMG (UDZO) named "Omni <version>".
hdiutil create -volname "$NAME $VERSION" -srcfolder "$STAGE" -fs HFS+ -format UDZO -ov "$DMG" >/dev/null

echo "Created $DMG ($(du -h "$DMG" | cut -f1))"
