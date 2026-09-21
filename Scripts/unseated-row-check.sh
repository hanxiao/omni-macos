#!/bin/bash
# ONE ROW WITH NO SLOT MUST NOT TAKE AN INDEX OFF THE AIR.
#
#   ./Scripts/unseated-row-check.sh <index-dir>
#
# Found on a real 28 GB index: a single chunk out of 10,540,581 carried `slot = -1`, the by-slot
# loader's "every row is seated" guard failed by exactly one, every repair below it was skipped,
# and the app showed "Omni can't open its index" quoting a bookkeeping number that had nothing to
# do with the cause. The row's 1536-byte vector was in `pending_vecs` the whole time.
#
# WHY THIS IS A SCRIPT AND NOT A UNIT TEST: the loader only runs on an index with a real coverage
# claim, and a claim only advances into the NAMED vector sidecar - which a small fixture never
# gets, whatever it is built with. `UnseatedRowTests` skips for exactly that reason and points
# here. This runs against an index big enough to be real.
#
# It only READS the index it is given. Point it at a COPY if you care about the original.
set -u
cd "$(dirname "$0")/.."
D=${1:-}
[ -n "$D" ] && [ -f "$D/index.sqlite" ] || { echo "usage: $0 <index-dir>"; exit 2; }
V=./.build/release/omni-verify
[ -x "$V" ] || { echo "missing $V - swift build -c release --product omni-verify"; exit 2; }

echo "=== unseated rows (slot < 0), and whether each still has a vector to place:"
sqlite3 -readonly "$D/index.sqlite" "
  SELECT 'unseated      ' || COUNT(*) FROM chunks WHERE slot < 0
  UNION ALL
  SELECT 'unplaceable   ' || COUNT(*) FROM chunks c WHERE c.slot < 0
     AND NOT EXISTS (SELECT 1 FROM pending_vecs p WHERE p.chunk_id = c.id);" 2>/dev/null \
  || sqlite3 -readonly "$D/index.sqlite" "
  SELECT 'unseated      ' || COUNT(*) FROM chunk WHERE slot < 0
  UNION ALL
  SELECT 'unplaceable   ' || COUNT(*) FROM chunk c WHERE c.slot < 0
     AND NOT EXISTS (SELECT 1 FROM pending_vecs p WHERE p.chunk_id = c.id);"

echo "=== does it open, and does it audit clean:"
"$V" storeaudit "$D/index.sqlite" 2>&1 | grep -E "placed [0-9]+ row|by-slot load declined|failing check|unreadable|rowTable"
