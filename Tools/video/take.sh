#!/bin/zsh
# take.sh <name> <seconds> <script> [app args...]: record one scene of the demo library
V=${VIDEO_DIR:-$(cd "$(dirname "$0")"; pwd)/work}
D=$V/demo; name=$1; secs=$2; script=$3; shift 3
BIN=$(cd "$(dirname "$0")/../.."; pwd)/.build/xcode-rel/Build/Products/Release/Omni.app/Contents/MacOS/Omni
pkill -x Omni; sleep 1
rm -rf $V/db-take $V/ocrcache; cp -R $V/db $V/db-take; mkdir -p $V/ocrcache $V/takes
ROOTS="(\"$D/Pictures\", \"$D/Documents\", \"$D/Downloads\")"
OMNI_PERF_LOG=1 OMNI_PERF_SCRIPT_SETTLE=0.3 OMNI_PERF_SCRIPT="$script" $BIN -omni.dbDir $V/db-take \
  -omni.addedFolders "$ROOTS" -omni.roots "$ROOTS" -omni.ephemeralUIState YES -omni.serving.enabled NO \
  -omni.ocr.cache.dir $V/ocrcache "$@" > $V/takes/$name.log 2>&1 &
pid=$!
$V/recwin $pid $secs $V/takes/$name.mov ${MINW:-1700} > $V/takes/$name.rec 2>&1
cat $V/takes/$name.rec | tail -1
kill $pid 2>/dev/null
