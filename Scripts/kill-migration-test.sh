#!/bin/bash
# SIGKILL DURING THE MIGRATION, then reopen and prove the index is still usable.
#
#   ./Scripts/kill-migration-test.sh 30 120 200
#
# One argument per run: how many seconds to let the migration go before killing it. Pick points
# that land in different phases - the slot backfill, then the fold - since they fail differently.
#
# Measured on a real 9,773,836-chunk v4 index, killed at 30s (early backfill), 120s (backfill at
# watermark 5,000,000) and 200s (inside the fold, 166,011 duplicates moved, the numbering already
# marked non-sequential). All three reopen with 0 failing audit checks and the identical search
# digest ba7a13400e714f79. The 200s case is the one that used to leave an index that would not
# open at all.
#
# A long migration is exactly when a user force-quits, closes the lid, or runs out of battery.
# Nothing here is a clean close: the process is killed outright, so the WAL has whatever it had,
# the vector file is mid-write, and the sidecars are stale. The next open has to cope.
SRC=/Volumes/han2tb/omni-index-backup-premigration
M=/Volumes/han2tb/ai-models/jinaai/jina-embeddings-v5-omni-nano-mlx
V=./.build/debug/omni-verify
for DELAY in "$@"; do
  W=/Volumes/han2tb/killtest
  rm -rf $W; mkdir -p $W
  for f in $SRC/*; do cp -c "$f" "$W/$(basename "$f")"; done
  echo "=== kill after ${DELAY}s"
  $V fold $W/index.sqlite > /tmp/kill-$DELAY.log 2>&1 &
  DRIVER=$!
  sleep "$DELAY"
  # The driver is a shell child; kill the omni-verify itself, hard.
  pkill -9 -f "omni-verify fold $W/index.sqlite" 2>/dev/null
  kill -9 $DRIVER 2>/dev/null
  wait $DRIVER 2>/dev/null
  sleep 2
  echo "  killed mid: $(tail -2 /tmp/kill-$DELAY.log | tr '\n' ' ' | cut -c1-90)"
  echo "  meta: $(sqlite3 -readonly $W/index.sqlite "select group_concat(key||'='||value,' ') from meta where key like '%fold%' or key like '%slot%';" 2>&1 | cut -c1-110)"
  echo "  --- reopen:"
  $V storeaudit $W/index.sqlite 2>&1 | grep -E "FAIL|failing check|unreadable|Fatal" | head -4
  $V searchreal $M $W/index.sqlite 3 2>&1 | grep -E "SEARCHREAL|wrong model|Fatal" | head -2
  rm -rf $W
done
