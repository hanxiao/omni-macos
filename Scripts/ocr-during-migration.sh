#!/bin/bash
# TRANSCRIBE A DOCUMENT WHILE A REAL v4 INDEX MIGRATES UNDERNEATH IT.
#
#   ./Scripts/ocr-during-migration.sh [pdf] [pages]
#
# The two halves of this app that each own a scarce resource, run at the same time: the OCR model
# saturates the GPU for a minute with 4.53 GB of weights resident, and the migration holds the
# store's serial queue and rewrites a 6 GB SQLite file beside a 22 GB vector file. They are
# supposed to be independent. "Supposed to be" is the class of claim this repository exists to
# stop making.
#
# WHY NOT XCUITest. There is a UI test for this shape and it cannot be trusted here: every
# XCUITest query first waits for the app to go idle, and an OCR run streaming at 24 Hz never does,
# so the workspace's own elements are invisible for the whole run - the shipped OCRWorkspaceUITests
# fails the same way on a tiny index with no migration at all, while the same build transcribes the
# page perfectly when driven by hand (verified in Release and in Debug). A headless pair of
# processes measures the contention exactly and yields a digest that can be compared.
#
# WHAT IS ASSERTED: the transcript is byte-identical to a solo run (the OCR digest), the migration
# finishes with a clean audit, and the search digest is the one the pre-migration index answered.
set -u
cd "$(dirname "$0")/.."
PDF=${1:-$HOME/Documents/2506.18902v3.pdf}
PAGES=${2:-8}
SRC=${OMNI_OCRMIG_SRC:-/Volumes/han2tb/omni-index-backup-premigration}
W="$(dirname "$SRC")/omni-ocrmig"
OCRM=${OMNI_OCR_MODEL:-$HOME/Library/Application Support/Omni/jina-ocr-v1-q8-mtp-mlx}
M=${OMNI_MODEL_DIR:-/Volumes/han2tb/ai-models/jinaai/jina-embeddings-v5-omni-nano-mlx}
V=./.build/release/omni-verify
T=./.build/release/opentime
O=./.build/release/ocr-verify
export OMNI_FREE_LIST=1

for f in "$V" "$T" "$O"; do
  [ -x "$f" ] || { echo "missing $f - swift build -c release --product omni-verify --product opentime --product ocr-verify"; exit 2; }
done
[ -f "$PDF" ] || { echo "no pdf at $PDF"; exit 2; }
[ -d "$OCRM" ] || { echo "OCR model not installed at $OCRM"; exit 0; }
[ -f "$SRC/index.sqlite" ] || { echo "no index at $SRC"; exit 2; }

echo "=== solo OCR baseline"
"$O" "$OCRM" --pdf "$PDF" --pages "$PAGES" --batch 8 2>&1 | grep -E "pages in|digest"
SOLO=$("$O" "$OCRM" --pdf "$PDF" --pages "$PAGES" --batch 8 2>&1 | grep "document digest" | awk '{print $3}')
echo "    solo digest $SOLO"

rm -rf "$W"; mkdir -p "$W"
for f in "$SRC"/*; do cp -c "$f" "$W/$(basename "$f")"; done
echo "=== migration and OCR at the same time"
"$T" "$W/index.sqlite" migrate > /tmp/ocrmig-migrate.log 2>&1 &
MIG=$!
sleep 5                     # let the migration get past open and into the backfill
"$O" "$OCRM" --pdf "$PDF" --pages "$PAGES" --batch 8 > /tmp/ocrmig-ocr.log 2>&1
grep -E "pages in|digest" /tmp/ocrmig-ocr.log
UNDER=$(grep "document digest" /tmp/ocrmig-ocr.log | awk '{print $3}')
wait $MIG; MIGRC=$?
tail -3 /tmp/ocrmig-migrate.log

rc=0
if [ "$SOLO" != "$UNDER" ]; then
  echo "=== FAIL: the transcript changed under the migration: $SOLO -> $UNDER"; rc=1
else
  echo "=== the transcript is identical under the migration ($SOLO)"
fi
[ "$MIGRC" = "0" ] || { echo "=== FAIL: the migration did not complete (rc=$MIGRC)"; rc=1; }

echo "=== the index afterwards"
"$V" storeaudit "$W/index.sqlite" 2>&1 | grep -E "rowTable|failing check|FAIL"
[ -d "$M" ] && "$V" searchreal "$M" "$W/index.sqlite" 5 2>&1 | grep -E "SEARCHREAL|wrong model"
echo "    (the pre-migration index answers digest=ba7a13400e714f79)"
[ "${OMNI_KEEP_CLONE:-0}" = "1" ] || rm -rf "$W"
exit $rc
