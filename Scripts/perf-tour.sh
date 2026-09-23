#!/bin/zsh
# Drive UITests/PerfTourUITests against a prebuilt index and record what it cost.
#
#   ./Scripts/perf-tour.sh <corpus|-> <index-dir> <out-dir> [reserve-dir]
#
# `-` for the corpus tours the app's own folders (a real index). OMNI_PERF_APP=<.app> launches that
# build instead of the test target; OMNI_PERF_ARGS="-k|v|..." adds launch arguments;
# OMNI_PERF_QUERIES="a|b" and OMNI_PERF_BROWSE="/dir:/dir" replace the corpus's queries and folders.
#
# <index-dir> is COPIED before the run and never opened itself, so every run starts from the same
# index. With [reserve-dir], its files are copied into the corpus 5 s after launch - the app indexes
# them while the tour searches - and removed again afterwards. Needs automation mode
# (Scripts/automation-window.sh) and no other Omni running; it quits one if it finds it.
#
# Writes to <out-dir>: tour.log (PHASE lines), hang.log (main-thread blocks), stderr.log (perf
# log), footprint.log (epoch + phys_footprint every 2 s).
set -u
cd "$(dirname "$0")/.."
CORPUS=$1; BASE=$2; OUT=$3; RESERVE=${4:-}
# A missing index is not an error the app reports: it opens an empty one, every search returns
# nothing, and the tour measures an app with no rows to draw.
[ -f "$BASE/index.sqlite" ] || { echo "no index at $BASE"; exit 2; }
mkdir -p "$OUT"; rm -f "$OUT"/*.log(N); true
osascript -e 'tell application id "io.hanxiao.omni" to quit' >/dev/null 2>&1
for _ in $(seq 1 30); do pgrep -x Omni >/dev/null || break; sleep 1; done
# A failed XCUITest run leaves its app behind, and one too busy to answer the quit above would
# otherwise run alongside the next launch on an index this script is about to delete and re-clone.
pkill -TERM -x Omni 2>/dev/null && sleep 3
DB="$OUT/db"; [ "$CORPUS" = "-" ] && { DB="$BASE.run"; CORPUS=""; }
rm -rf "$DB"; cp -c -R "$BASE" "$DB" 2>/dev/null || cp -R "$BASE" "$DB"   # an APFS clone when it can

(
  while true; do
    pid=$(pgrep -x Omni | head -1)
    [ -n "$pid" ] && echo "$(date +%s) $(footprint "$pid" 2>/dev/null | awk '/phys_footprint:/{print $2$3}')" >> "$OUT/footprint.log"
    sleep 2
  done
) &
SAMPLER=$!

# OMNI_PERF_SAMPLE="10:35 110:80" runs `sample` (every thread's stacks) for 35 s starting 10 s after
# launch, and for 80 s from 110 s: the call trees that say what a stall in hang.log was doing.
for w in ${=OMNI_PERF_SAMPLE:-}; do
  ( until pgrep -x Omni >/dev/null; do sleep 0.5; done; sleep ${w%%:*}
    sample "$(pgrep -x Omni | head -1)" ${w##*:} -mayDie -file "$OUT/sample-${w%%:*}.txt" >/dev/null 2>&1 ) &
done

if [ -n "$RESERVE" ]; then
  ( until pgrep -x Omni >/dev/null; do sleep 1; done; sleep 5
    echo "$(date +%s) reserve copy start" >> "$OUT/footprint.log"
    cp -R "$RESERVE"/. "$CORPUS"/
    echo "$(date +%s) reserve copy done" >> "$OUT/footprint.log" ) &
fi

[ -n "${OMNI_PERF_QUERIES:-}" ] && export TEST_RUNNER_OMNI_PERF_QUERIES="$OMNI_PERF_QUERIES"
[ -n "${OMNI_PERF_BROWSE:-}" ] && export TEST_RUNNER_OMNI_PERF_BROWSE="$OMNI_PERF_BROWSE"
# On the app's own folders the build under test runs as the user's Omni: it advances the event
# checkpoint and may write other settings. Snapshot them and put them back afterwards.
PREFS=""
if [ -z "$CORPUS" ]; then PREFS="$OUT/prefs-before.plist"; defaults export io.hanxiao.omni "$PREFS"; fi

TEST_RUNNER_OMNI_PERF_DB="$DB" TEST_RUNNER_OMNI_PERF_CORPUS="$CORPUS" \
TEST_RUNNER_OMNI_PERF_HANG="$OUT/hang.log" TEST_RUNNER_OMNI_PERF_STDERR="$OUT/stderr.log" \
TEST_RUNNER_OMNI_PERF_HANG_MS="${OMNI_PERF_HANG_MS:-50}" TEST_RUNNER_OMNI_PERF_PHASES="${OMNI_PERF_PHASES:-}" \
TEST_RUNNER_OMNI_PERF_APP="${OMNI_PERF_APP:-}" TEST_RUNNER_OMNI_PERF_ARGS="${OMNI_PERF_ARGS:-}" \
TEST_RUNNER_OMNI_PERF_APPENV="${OMNI_PERF_APPENV:-}" \
OMNI_UI_CONFIG=Release OMNI_DD=${OMNI_DD:-.build/xcode-perf} \
  ./Scripts/ui-test.sh PerfTourUITests/testTour > "$OUT/xcodebuild.log" 2>&1
RC=$?
kill $SAMPLER 2>/dev/null
for _ in $(seq 1 30); do pgrep -x Omni >/dev/null || break; sleep 1; done
if [ -n "$PREFS" ]; then defaults import io.hanxiao.omni "$PREFS" && echo "prefs restored"; fi
grep -E "^PHASE " "$OUT/xcodebuild.log" > "$OUT/tour.log"

if [ -n "$RESERVE" ]; then
  # Undo the churn so the next run starts from the same corpus.
  (cd "$RESERVE" && find . -type f) | while IFS= read -r f; do rm -f "$CORPUS/$f"; done
  find "$CORPUS" -type d -empty -delete
fi
echo "rc=$RC; $(wc -l < "$OUT/tour.log") phases, $(grep -c blocked "$OUT/hang.log" 2>/dev/null) blocks"
exit $RC
