#!/bin/zsh
# Run UITests/FullChaosUITests against a prebuilt index and collect everything that says what broke.
#
#   ./Scripts/chaos-run.sh <corpus> <index-dir> <out-dir> [reserve-dir]
#
# OMNI_CHAOS_SEED (default: the clock) and OMNI_CHAOS_SECONDS (600) steer the run; the seed is printed
# as `CHAOS seed` in xcodebuild.log and replays the same actions. With [reserve-dir], batches of its
# files are copied into the corpus and removed again all run long, so the indexer is always writing.
#
# Writes to <out-dir>: xcodebuild.log (the CHAOS action trail and test failures), hang.log (main-thread
# stalls over 250 ms), stderr.log (the app's perf log), oslog.log (errors and faults the app logged,
# SwiftUI runtime warnings included), footprint.log, and crashes/ (any Omni crash report).
set -u
cd "$(dirname "$0")/.."
CORPUS=$1; BASE=$2; OUT=$3; RESERVE=${4:-}
[ -f "$BASE/index.sqlite" ] || { echo "no index at $BASE"; exit 2; }
mkdir -p "$OUT/crashes"; rm -f "$OUT"/*.log(N)
osascript -e 'tell application id "io.hanxiao.omni" to quit' >/dev/null 2>&1
for _ in $(seq 1 30); do pgrep -x Omni >/dev/null || break; sleep 1; done
pkill -TERM -x Omni 2>/dev/null && sleep 3
DB="$OUT/db"; rm -rf "$DB"; cp -c -R "$BASE" "$DB" 2>/dev/null || cp -R "$BASE" "$DB"
touch "$OUT/.started"

/usr/bin/log stream --style compact --predicate 'process == "Omni" && (messageType == error || messageType == fault)' \
  > "$OUT/oslog.log" 2>/dev/null &
OSLOG=$!
( while true; do
    pid=$(pgrep -x Omni | head -1)
    [ -n "$pid" ] && echo "$(date +%s) $(footprint "$pid" 2>/dev/null | awk '/phys_footprint:/{print $2$3}')" >> "$OUT/footprint.log"
    sleep 5
  done ) &
SAMPLER=$!
# STUCK DETECTOR. The stall log is written when a block ENDS, which for a real hang is long after
# anyone could look. So when the action trail goes quiet for 20 s, sample the app while it is stuck.
( last=""; quiet=0
  while true; do
    sleep 5
    cur=$(grep -c '^CHAOS [0-9]' "$OUT/xcodebuild.log" 2>/dev/null)
    if [ "$cur" = "$last" ]; then quiet=$((quiet + 5)); else quiet=0; last=$cur; fi
    if [ $quiet -eq 20 ] && pid=$(pgrep -x Omni | head -1); then
      sample "$pid" 5 -file "$OUT/stuck-$cur.txt" >/dev/null 2>&1
    fi
  done ) &
STUCK=$!
CHURN=""
if [ -n "$RESERVE" ]; then
  ( cd "$RESERVE"; files=(${(f)"$(find . -type f | sort)"}); i=1
    while true; do
      sleep 20
      batch=(${files[$i,$((i+149))]}); i=$(( i + 150 )); (( i > ${#files} )) && i=1
      for f in $batch; do mkdir -p "$CORPUS/${f:h}"; cp "$f" "$CORPUS/$f" 2>/dev/null; done
      sleep 20
      for f in $batch; do rm -f "$CORPUS/$f"; done
    done ) &
  CHURN=$!
fi

TEST_RUNNER_OMNI_PERF_DB="$DB" TEST_RUNNER_OMNI_PERF_CORPUS="$CORPUS" \
TEST_RUNNER_OMNI_PERF_HANG="$OUT/hang.log" TEST_RUNNER_OMNI_PERF_STDERR="$OUT/stderr.log" \
TEST_RUNNER_OMNI_CHAOS_SEED="${OMNI_CHAOS_SEED:-}" TEST_RUNNER_OMNI_CHAOS_SECONDS="${OMNI_CHAOS_SECONDS:-600}" \
OMNI_UI_CONFIG=Release OMNI_DD=${OMNI_DD:-.build/xcode-perf} \
  ./Scripts/ui-test.sh FullChaosUITests/testChaos > "$OUT/xcodebuild.log" 2>&1
RC=$?
kill $SAMPLER $OSLOG $STUCK 2>/dev/null
[ -n "$CHURN" ] && { kill $CHURN 2>/dev/null; (cd "$RESERVE" && find . -type f) | while IFS= read -r f; do rm -f "$CORPUS/$f"; done
                     find "$CORPUS" -type d -empty -delete; }
pkill -TERM -x Omni 2>/dev/null
find ~/Library/Logs/DiagnosticReports -name 'Omni*' -newer "$OUT/.started" -exec cp {} "$OUT/crashes/" \; 2>/dev/null
echo "rc=$RC seed=$(grep -m1 -o 'CHAOS seed [0-9]*' "$OUT/xcodebuild.log") actions=$(grep -c '^CHAOS [0-9]' "$OUT/xcodebuild.log")" \
     "failures=$(grep -c ': error: ' "$OUT/xcodebuild.log") stalls=$(grep -c blocked "$OUT/hang.log" 2>/dev/null)" \
     "crashes=$(ls "$OUT/crashes" | wc -l | tr -d ' ') oslog=$(grep -vc '^Filtering\|^Timestamp' "$OUT/oslog.log")"
exit $RC
