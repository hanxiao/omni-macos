#!/bin/bash
# Drive the app chaotically while a REAL v4 index migrates underneath it.
#
#   ./Scripts/migration-chaos.sh [source-index-dir]
#
# The clone has to happen here rather than inside the test: `cp -c` only clones within one APFS
# volume, so the scratch must sit beside the source, and the XCUITest runner is sandboxed and may
# not write under /Volumes at all. So this clones (instant, no space) and hands the path in.
set -e
# Any index directory: a v4 one exercises the migration, a fully migrated one exercises the
# steady state. Both are worth running.
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
# The app's own stderr, which XCUITest otherwise swallows.
LOGF=${OMNI_CHAOS_LOG:-/tmp/omni-chaos-stderr.log}
: > "$LOGF"
export TEST_RUNNER_OMNI_MIGCHAOS_STDERR="$LOGF"
export OMNI_MIGCHAOS_STDERR="$LOGF"
./Scripts/ui-test.sh MigrationChaosUITests
rc=$?
echo "=== app stderr: $(grep -c "" "$LOGF" 2>/dev/null || echo 0) lines"
grep -inE "error|fail|warn|refus|abandon|unreadable|corrupt|cannot|invalid" "$LOGF" 2>/dev/null | head -40
rm -rf "$W"
exit $rc
