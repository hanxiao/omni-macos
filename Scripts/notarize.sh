#!/bin/bash
# Notarize a Developer-ID-signed Omni.app and produce a stapled, notarized DMG.
#
#   ./Scripts/notarize.sh <notary-profile> [path/to/Omni.app] [version]
#
# Prerequisites (one-time):
#   1. A "Developer ID Application" certificate in the login keychain (Xcode > Settings >
#      Accounts > Manage Certificates > + > Developer ID Application).
#   2. Notary credentials stored under a keychain profile name, e.g.:
#        xcrun notarytool store-credentials omni-notary \
#          --key   AuthKey_XXXX.p8 \
#          --key-id  <KEY_ID> \
#          --issuer  <ISSUER_ID>
#      (or the Apple-ID variant: --apple-id <id> --team-id MTECXQ97E6 --password <app-specific>)
#
# The app MUST already be signed with Developer ID + hardened runtime (build it with
# ./Scripts/build-app.sh Release CODE_SIGN_IDENTITY="Developer ID Application" ...). This script
# does not re-sign; it only verifies, notarizes, staples, and packages.
set -euo pipefail
cd "$(dirname "$0")/.."

PROFILE="${1:?usage: notarize.sh <notary-profile> [app] [version]}"
APP="${2:-.build/xcode-rel/Build/Products/Release/Omni.app}"
[ -d "$APP" ] || { echo "app not found: $APP"; exit 1; }
VERSION="${3:-$(/usr/libexec/PlistBuddy -c 'Print CFBundleShortVersionString' "$APP/Contents/Info.plist")}"

echo "==> Verifying the app is a clean Developer ID distribution build"
AUTH=$(codesign -dvv "$APP" 2>&1 | grep -m1 '^Authority=' || true)
echo "    $AUTH"
case "$AUTH" in
  *"Developer ID Application"*) ;;
  *) echo "ERROR: app is not signed with 'Developer ID Application' (got: $AUTH).
   Build it with: ./Scripts/build-app.sh Release CODE_SIGN_IDENTITY=\"Developer ID Application\" \\
     CODE_SIGN_STYLE=Manual PROVISIONING_PROFILE_SPECIFIER= OTHER_CODE_SIGN_FLAGS=--timestamp"; exit 1 ;;
esac
if codesign -d --entitlements - "$APP" 2>/dev/null | grep -q 'get-task-allow'; then
  echo "ERROR: app carries com.apple.security.get-task-allow (a debug entitlement); notarization
   will reject it. This means it was signed with 'Apple Development', not 'Developer ID Application'."; exit 1
fi
codesign --verify --deep --strict --verbose=2 "$APP"

echo "==> Notarizing the app"
ZIP="$(mktemp -d)/Omni.zip"
/usr/bin/ditto -c -k --keepParent "$APP" "$ZIP"
xcrun notarytool submit "$ZIP" --keychain-profile "$PROFILE" --wait
echo "==> Stapling the app"
xcrun stapler staple "$APP"
xcrun stapler validate "$APP"

echo "==> Packaging the DMG (no Gatekeeper-warning note: this build is notarized)"
NOTARIZED=1 ./Scripts/make_dmg.sh "$APP" "$VERSION"
DMG="dist/Omni-${VERSION}.dmg"

# Sign the disk image itself with Developer ID before notarizing. The stapled ticket alone is not
# enough for `spctl --type open` - Gatekeeper also wants a Developer ID signature on the DMG.
echo "==> Signing the DMG with Developer ID"
codesign --force --sign "Developer ID Application" --timestamp "$DMG"

echo "==> Notarizing and stapling the DMG"
xcrun notarytool submit "$DMG" --keychain-profile "$PROFILE" --wait
xcrun stapler staple "$DMG"

echo "==> Final Gatekeeper verdict"
spctl --assess --type open --context context:primary-signature -v "$DMG" || true
spctl --assess --type execute -vv "$APP" || true
echo "Done: $DMG (notarized + stapled)"
