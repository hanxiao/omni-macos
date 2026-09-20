#!/bin/bash
# SIGKILL DURING THE SPLIT MIGRATION, then reopen and prove the index is still usable.
#
#   ./Scripts/kill-split-migration.sh 30 120 240
#
# One argument per run: how many seconds to let the migration go before killing it. Pick points
# that land in DIFFERENT PHASES, because they fail differently and a run that only ever lands in
# one of them proves the least interesting case:
#
#   ~30s    the slot backfill, sliced, watermark moving
#   ~120s   coverage advancing, blobs being cleared by position range
#   ~240s   inside the off-queue split build, with `chunk` and `occurrence` half written
#
# THE 240s CASE IS THE ONE THIS SCRIPT EXISTS FOR. The build writes millions of rows on a second
# connection and publishes one meta flag at the end; killed before that flag, the next open must
# see a v4 index with some orphaned split rows and rebuild them, NOT a half-built split it
# believes. Killed after it, the loader switches models and the 3.77M positions the split freed
# have to be recorded on that first open or `coverageAudit` calls every one of them breakage.
#
# `Scripts/kill-migration-test.sh` is the same test for the FOLD, which the split replaces. Both
# are kept while both paths exist.
#
# Nothing here is a clean close: the process is killed outright, so the WAL has whatever it had,
# the vector file is mid-write, and the sidecars are stale. The next open has to cope. That is
# the point - a long migration is exactly when a user force-quits, closes the lid, or runs out
# of battery.
set -u
cd "$(dirname "$0")/.."
SRC=${OMNI_KILL_SRC:-/Volumes/han2tb/omni-index-backup-premigration}
M=${OMNI_MODEL_DIR:-/Volumes/han2tb/ai-models/jinaai/jina-embeddings-v5-omni-nano-mlx}
W=${OMNI_KILL_WORK:-/Volumes/han2tb/killsplit}
V=./.build/release/omni-verify
T=./.build/release/opentime
export OMNI_FREE_LIST=1

for f in "$V" "$T"; do
  [ -x "$f" ] || { echo "missing $f - swift build -c release --product omni-verify --product opentime"; exit 2; }
done
[ -d "$SRC" ] || { echo "no source index at $SRC"; exit 2; }

for DELAY in "$@"; do
  rm -rf "$W"; mkdir -p "$W"
  # -c is an APFS clone: the 28 GB copy costs no space and no time until something writes.
  for f in "$SRC"/*; do cp -c "$f" "$W/$(basename "$f")"; done
  echo "=== kill after ${DELAY}s"
  "$T" "$W/index.sqlite" migrate > "/tmp/killsplit-$DELAY.log" 2>&1 &
  DRIVER=$!
  sleep "$DELAY"
  # The driver is a shell child; kill the opentime itself, hard, and its off-queue build with it.
  pkill -9 -f "opentime $W/index.sqlite" 2>/dev/null
  kill -9 $DRIVER 2>/dev/null
  wait $DRIVER 2>/dev/null
  sleep 2
  echo "  killed mid: $(tail -2 "/tmp/killsplit-$DELAY.log" | tr '\n' ' ' | cut -c1-100)"
  echo "  meta: $(sqlite3 -readonly "$W/index.sqlite" \
      "select group_concat(key||'='||value,' ') from meta where key like '%split%' or key like '%slot%' or key like 'vecs%';" 2>&1 | cut -c1-140)"
  echo "  tables: $(sqlite3 -readonly "$W/index.sqlite" \
      "select 'chunk='||(select count(*) from chunk)||' occ='||(select count(*) from occurrence)||' free='||(select count(*) from free_slot);" 2>&1 | cut -c1-90)"
  echo "  --- reopen:"
  "$V" storeaudit "$W/index.sqlite" 2>&1 | grep -E "FAIL|failing check|rowTable|unreadable|Fatal" | head -5
  "$V" searchreal "$M" "$W/index.sqlite" 3 2>&1 | grep -E "SEARCHREAL|wrong model|Fatal" | head -2
  rm -rf "$W"
done
