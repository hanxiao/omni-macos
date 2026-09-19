#!/bin/bash
# Drive the app chaotically while a REAL v4 index migrates underneath it.
#
#   ./Scripts/migration-chaos.sh [source-index-dir]
#
# The clone has to happen here rather than inside the test: `cp -c` only clones within one APFS
# volume, so the scratch must sit beside the source, and the XCUITest runner is sandboxed and may
# not write under /Volumes at all. So this clones (instant, no space) and hands the path in.
set -e
SRC=${1:-/Volumes/han2tb/omni-index-backup-premigration}
[ -f "$SRC/index.sqlite" ] || { echo "no index at $SRC"; exit 1; }
W="$(dirname "$SRC")/omni-migchaos-live"
rm -rf "$W"; mkdir -p "$W"
for f in "$SRC"/*; do cp -c "$f" "$W/"; done
echo "cloned $(ls -la "$W/index.sqlite" | awk '{printf "%.2f GB", $5/1073741824}') to $W"
pkill -x Omni 2>/dev/null || true; sleep 2
# TEST_RUNNER_ is the only prefix xcodebuild forwards into the test runner's environment.
export TEST_RUNNER_OMNI_MIGCHAOS_DB="$W"
export OMNI_MIGCHAOS_DB="$W"
./Scripts/ui-test.sh MigrationChaosUITests
rc=$?
rm -rf "$W"
exit $rc
