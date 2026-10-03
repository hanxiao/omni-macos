#!/bin/zsh
# Build Tools/dropprobe into <dest>/DropProbe.app (default .build/DropProbe.app): the drag
# destination FileDragUITests drops onto. Ad-hoc signed; it only ever runs on this machine.
set -euo pipefail
cd "$(dirname "$0")/.."
DEST=${1:-.build}
APP="$DEST/DropProbe.app"
rm -rf "$APP"; mkdir -p "$APP/Contents/MacOS"
swiftc -O -o "$APP/Contents/MacOS/DropProbe" Tools/dropprobe/DropProbe.swift
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleIdentifier</key><string>io.hanxiao.omni.dropprobe</string>
<key>CFBundleName</key><string>DropProbe</string>
<key>CFBundleExecutable</key><string>DropProbe</string>
<key>CFBundlePackageType</key><string>APPL</string>
<key>NSPrincipalClass</key><string>NSApplication</string>
</dict></plist>
PLIST
codesign --force --sign - "$APP"
echo "drop probe: $APP"
