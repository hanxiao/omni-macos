#!/bin/zsh
# probe.sh <name> "<steps>": the Developer-ID dev build on the film's index clone, one screenshot.
# Never touches the installed Omni: it is found and quit by its own pid only.
H=${0:A:h}; W=$H/../work/launch; APP=$H/../../../.build/scratch/live/Omni.app
NAME=$1; STEPS=$2
OMNI_PERF_LOG=1 OMNI_PERF_SCRIPT_SETTLE=0.3 OMNI_PERF_SCRIPT="frame:${FRAME:-1600x1000};$STEPS;wait:30" "$APP/Contents/MacOS/Omni" \
  -NSAppSleepDisabled YES -omni.dbDir /Volumes/han2tb/launch-film-index -omni.serving.enabled NO \
  -omni.searchHistory "<$(cat $W/takes/hist.hex)>" -omni.ocr.cache.dir $W/takes/ocrcache > $W/takes/$NAME.log 2>&1 &
PID=$!
LAST=${STEPS##*;}
for i in $(seq 1 120); do grep -qF "script ${LAST%%:*}" $W/takes/$NAME.log && break; sleep 0.5; done
sleep ${SHOT_DELAY:-3}
$H/../../../.build/scratch/stall/winshot $PID $W/takes/$NAME.png > /dev/null
kill $PID; wait $PID 2>/dev/null
