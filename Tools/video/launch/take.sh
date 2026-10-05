#!/bin/zsh
# take.sh <name> <seconds> "<steps>": record one app take on the film's index clone with ScreenCaptureKit.
# The installed Omni is never touched: this instance is started here and killed by its own pid.
H=${0:A:h}; W=$H/../work/launch; APP=$H/../../../.build/scratch/live/Omni.app
NAME=$1; SECS=$2; STEPS=$3
OMNI_PERF_LOG=1 OMNI_PERF_SCRIPT_SETTLE=0.3 OMNI_PERF_SCRIPT="frame:${FRAME:-1280x800};$STEPS;wait:60" "$APP/Contents/MacOS/Omni" \
  -NSAppSleepDisabled YES -omni.dbDir /Volumes/han2tb/launch-film-index -omni.serving.enabled NO \
  -omni.searchHistory "<$(cat $W/takes/hist.hex)>" -omni.ocr.cache.dir $W/takes/ocrcache > $W/takes/$NAME.log 2>&1 &
PID=$!
# record only once the index is loaded and the window has its final size
for i in $(seq 1 240); do grep -q "launch ready" $W/takes/$NAME.log && grep -qF "script frame" $W/takes/$NAME.log && break; sleep 0.25; done
sleep ${PRE:-0.5}
$W/recwin $PID $SECS $W/takes/$NAME.mov 1000 > $W/takes/$NAME.rec 2>&1
kill $PID 2>/dev/null; wait $PID 2>/dev/null
tail -1 $W/takes/$NAME.rec
