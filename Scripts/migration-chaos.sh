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
# Feature flags through to the app under test, same prefix rule.
for k in OMNI_CHUNK_SPLIT OMNI_FREE_LIST OMNI_SPLIT_CUTOVER; do
  v=$(eval echo \$$k); [ -n "$v" ] && export TEST_RUNNER_$k="$v"
done
for k in OMNI_MIGCHAOS_QUIET_SECONDS; do
  v=$(eval echo \$$k); [ -n "$v" ] && export TEST_RUNNER_$k="$v"
done
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

# WHAT THE RUN ACTUALLY EXERCISED, read off the index before it is thrown away.
#
# A chaos run that passes proves nothing about a feature the run never switched on, and both
# split flags are gated on `chunk_split_backfilled` being set - which happens from the coverage
# stamp, which YIELDS TO SEARCHES, which is the one thing this suite does continuously. So the
# arm can be on, the test can pass, and the split can have sat untouched the whole time. That is
# exactly how OMNI_CHUNK_SPLIT and OMNI_SPLIT_CUTOVER both reported green for weeks.
echo "=== what the index ended up as"
sqlite3 -readonly "$W/index.sqlite" "
  SELECT 'chunks      ' || COUNT(*) FROM chunks
  UNION ALL SELECT 'chunk_text  ' || COUNT(*) FROM chunk_text
  UNION ALL SELECT 'chunk       ' || COUNT(*) FROM chunk
  UNION ALL SELECT 'occurrence  ' || COUNT(*) FROM occurrence
  UNION ALL SELECT 'snippet     ' || COUNT(*) FROM chunk_snippet
  UNION ALL SELECT 'free_slot   ' || COUNT(*) FROM free_slot
  UNION ALL SELECT 'vec_holes   ' || COUNT(*) FROM vec_holes;" 2>&1
echo "=== migration markers"
sqlite3 -readonly "$W/index.sqlite" \
  "SELECT key || '=' || value FROM meta WHERE key LIKE 'chunk_%' OR key LIKE 'vecs_%' ORDER BY key;" 2>&1
if [ "${OMNI_CHUNK_SPLIT:-0}" = "1" ]; then
  built=$(sqlite3 -readonly "$W/index.sqlite" \
    "SELECT COALESCE((SELECT CAST(value AS INTEGER) FROM meta WHERE key='chunk_split_backfilled'),0);" 2>/dev/null)
  if [ "$built" != "1" ]; then
    echo "=== WARNING: OMNI_CHUNK_SPLIT=1 but the split was never built in this run."
    echo "    The run exercised v4. Give it longer, or drive fewer searches, before believing it."
    rc=2
  else
    echo "=== the split WAS built and in use for this run"
  fi
fi
[ "${OMNI_KEEP_CLONE:-0}" = "1" ] || rm -rf "$W"
exit $rc
