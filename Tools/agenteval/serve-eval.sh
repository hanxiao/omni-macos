#!/bin/zsh
# Start an ISOLATED Omni serving the eval testbed, for agenteval.py.
#
# A fresh APFS clone of the testbed's nano index, the testbed tree as the only folder, port 51399
# (51299 is a real install's), its own CLI socket, no Linked Macs on the network, no clipboard
# history, no uploads, App Nap off, and a footprint watchdog that kills it past the cap. Waits
# until it answers and has nothing left to index, then prints the pid. Stop it with `kill <pid>`.
#
#   Tools/agenteval/serve-eval.sh <bed dir> <new scratch dir> [app bundle] [cap GB]
set -euo pipefail
BED=${1:?bed dir}; SCRATCH=${2:?scratch dir (must not exist)}
APP=${3:-.build/xcode-rel/Build/Products/Release/Omni.app}
CAP_GB=${4:-24}
PORT=51399
[[ -e $SCRATCH ]] && { echo "$SCRATCH exists; give a new scratch dir so the index clone is fresh" >&2; exit 1; }
INDEX=$(ls -d "$BED"/index-nano-768-* | grep -v -- '-[a-z]*$' | head -1)
[[ -d $INDEX ]] || { echo "no nano index under $BED (run omni-verify eval first)" >&2; exit 1; }
curl -s -m 1 "http://127.0.0.1:$PORT/health" >/dev/null && { echo "something already serves $PORT" >&2; exit 1; }
mkdir -p "$SCRATCH"
cp -c -R "$INDEX" "$SCRATCH/index"

"$APP/Contents/MacOS/Omni" \
  -omni.dbDir "$SCRATCH/index" \
  -omni.addedFolders "(\"$BED/tree\")" \
  -omni.modelDir "$HOME/Library/Application Support/Omni/nano" \
  -omni.ephemeralUIState YES \
  -omni.serving.enabled YES -omni.serving.port $PORT \
  -omni.socketPath "$SCRATCH/omni.sock" \
  -omni.link.bonjour NO -omni.link.port 51499 -omni.link.name agenteval \
  -omni.clipboard.enabled NO \
  -omni.ocr.cache.dir "$SCRATCH/ocr" \
  -omni.profiling.uploadEnabled NO \
  -NSAppSleepDisabled YES \
  > "$SCRATCH/omni.log" 2>&1 &
PID=$!
echo $PID > "$SCRATCH/pid"

# The watchdog: an isolated instance left running was once seen at 100 GB.
(
  while kill -0 $PID 2>/dev/null; do
    kb=$(footprint -p $PID 2>/dev/null | awk '/Footprint:/ { v=$(NF-5); u=$(NF-4); m=(u=="GB")?1048576:(u=="MB")?1024:1; print int(v*m); exit }')
    if [[ -n $kb ]]; then
      echo "$(date +%s) $kb" >> "$SCRATCH/footprint.log"
      if (( kb > CAP_GB * 1048576 )); then echo "footprint ${kb} KB over ${CAP_GB} GB: killing $PID" >> "$SCRATCH/omni.log"; kill -9 $PID; fi
    fi
    sleep 2
  done
) </dev/null >/dev/null 2>&1 &

for i in {1..120}; do
  curl -s -m 1 "http://127.0.0.1:$PORT/health" >/dev/null && break
  sleep 1
done
curl -s -m 2 "http://127.0.0.1:$PORT/health" >/dev/null || { echo "Omni did not come up; see $SCRATCH/omni.log" >&2; kill $PID; exit 1; }
# Settled: list_sources reports no folder still indexing.
for i in {1..600}; do
  state=$(curl -s -m 5 -X POST "http://127.0.0.1:$PORT/mcp" -H 'Content-Type: application/json' \
    -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"list_sources","arguments":{}}}')
  [[ $state == *indexing* || $state == *loading* ]] || break
  sleep 2
done
echo $PID
