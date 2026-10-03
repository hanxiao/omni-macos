#!/bin/zsh
# Copy a built Omni.app to <dest> under bundle id io.hanxiao.omni.chaos, for Scripts/chaos-run.sh.
#
#   ./Scripts/chaos-app.sh [source.app] <dest.app>
#
# XCUITest's launch() terminates any running app with the target's bundle id, so a chaos run of the
# real bundle id quits the user's own Omni. The copy has its own id, and with it its own preferences
# domain, so it can run beside the user's app and cannot write the user's settings. Ad-hoc signed:
# it only ever runs on this machine.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ $# -eq 1 ]; then SRC=.build/xcode-rel/Build/Products/Release/Omni.app; DEST=$1; else SRC=$1; DEST=$2; fi
rm -rf "$DEST"
ditto "$SRC" "$DEST"
/usr/libexec/PlistBuddy -c "Set :CFBundleIdentifier io.hanxiao.omni.chaos" "$DEST/Contents/Info.plist"
/usr/libexec/PlistBuddy -c "Set :CFBundleName OmniChaos" "$DEST/Contents/Info.plist" 2>/dev/null || true
codesign -d --entitlements - --xml "$SRC" > /tmp/omni-chaos-ent.plist 2>/dev/null || true
if [ -s /tmp/omni-chaos-ent.plist ]; then
  codesign --force --deep --sign - --entitlements /tmp/omni-chaos-ent.plist "$DEST"
else
  codesign --force --deep --sign - "$DEST"
fi
echo "chaos app: $DEST ($(defaults read "$DEST/Contents/Info.plist" CFBundleIdentifier))"
