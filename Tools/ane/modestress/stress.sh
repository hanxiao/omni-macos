#!/bin/zsh
# ANE -> GPU -> ANE on one isolated instance, every surface exercised in each mode, every phase time-marked.
#   stress.sh <run dir> <model dir> <ane cache> [extra folder, e.g. HD clips] [app]
# Needs Tools/ane/powerlog built (swiftc -O Tools/ane/powerlog.swift -o Tools/ane/powerlog).
# Phases: index from empty (ANE) | idle | UI steps | served load (REST, MCP search + tag_image, CLI) | files added
# then the switch to GPU and the same, then back to ANE and the same. Logs: marks.log, gpu.log (per-queue GPU ns of
# the pid), fp.log (footprint), power.log (system W by unit), load.csv, app.log.
W=$1; MODEL=$2; CACHE=$3; HD=${4:-$1/drop}; B=${0:A:h}
APP=${5:-${B:h:h:h}/.build/xcode-rel/Build/Products/Release/Omni.app}; BED=/Volumes/han2tb/omni-eval/tree
rm -rf $W; mkdir -p $W/index $W/ocr $W/cue $W/drop
SOCK=/private/tmp/claude-501/ost-$$.sock; rm -f $SOCK
mark() { echo "$(date +%s.%N | cut -c1-14) $1" >> $W/marks.log; echo "[$(date +%T)] $1"; }
passes() { local n=$(grep -c "pass-done" $W/app.log 2>/dev/null); echo ${n:-0}; }
wait_log() {  # pattern, timeout s, min count
  local t0=$(date +%s); while [ $(( $(date +%s) - t0 )) -lt $2 ]; do
    local n=$(grep -c "$1" $W/app.log 2>/dev/null); [ ${n:-0} -ge ${3:-1} ] && return 0; sleep 1; done; mark "TIMEOUT waiting for $1"; return 1; }
wait_pass() {  # a pass-done after count $1, and nothing new for 8 s
  local t0=$(date +%s); while [ $(( $(date +%s) - t0 )) -lt ${2:-900} ]; do
    [ $(passes) -gt $1 ] && break; sleep 1; done
  sleep 8; }
UI="search:invoice from the dentist;search:a dog on the beach;type:budget meeting notes;search:zoo trip video;similar:$BED/Documents/Books/Sailing log 2025.pdf;map:$BED/Documents;wait:4;clear"
SCRIPT="waitfile:$W/cue/1;$UI;waitfile:$W/cue/2;set:ane=off;waitfile:$W/cue/3;$UI;waitfile:$W/cue/4;set:ane=on;waitfile:$W/cue/5;$UI;waitfile:$W/cue/6"
${B:h}/powerlog --stream 1 > $W/power.log &
PS=$!
mark launch-ane
OMNI_PERF_LOG=1 OMNI_PERF_SCRIPT="$SCRIPT" "$APP/Contents/MacOS/Omni" -omni.dbDir "$W/index" \
  -omni.addedFolders "(\"$BED\", \"$W/drop\", \"$HD\")" -omni.modelDir "$MODEL" -omni.ephemeralUIState YES \
  -omni.serving.enabled YES -omni.serving.port 51399 -omni.socketPath "$SOCK" \
  -omni.link.bonjour NO -omni.link.port 51499 -omni.link.name stress -omni.clipboard.enabled NO \
  -omni.ocr.cache.dir "$W/ocr" -omni.ane.cache.dir "$CACHE" -omni.profiling.uploadEnabled NO \
  -NSAppSleepDisabled YES -omni.computeUnit neuralEngine -omni.computeChoicePending NO -omni.update.lastCheckEpoch $(date +%s) > $W/app.log 2>&1 &
P=$!; echo $P > $W/pid
$B/sampler.sh $P $W &
CLI=$APP/Contents/Helpers/omni
load() { python3 $B/load.py --port 51399 --socket $SOCK --cli $CLI --phase $1 --seconds ${2:-45} --out $W/load.csv --images $BED/Pictures; }
add() {  # copy a batch of the bed's files (text, PDFs, images, audio) into the watched drop folder, plus salted notes
  mkdir -p $W/drop/$1; local n=0; : > $W/added-$1.txt
  for f in ${(f)"$(find $BED/Documents $BED/Pictures $BED/Recordings -type f \( -name "*.pdf" -o -name "*.txt" -o -name "*.md" -o -name "*.docx" -o -name "*.jpg" -o -name "*.png" -o -name "*.m4a" -o -name "*.heic" \) 2>/dev/null | sort | awk -v s=$2 'NR % 7 == s' | head -24)"}; do
    local d="$W/drop/$1/$1-$(basename "$f")"; cp "$f" "$d"; echo "$d" >> $W/added-$1.txt
    n=$((n+1)); echo "salt $1 $RANDOM $(date +%s%N) a note about sailing number $n" > "$W/drop/$1/$1-salt-$n.txt"; echo "$W/drop/$1/$1-salt-$n.txt" >> $W/added-$1.txt; done; }
wait_indexed() {  # every file added in batch $1 indexed and up to date, as the CLI reports it (10 min at most)
  local t0=$(date +%s)
  while [ $(( $(date +%s) - t0 )) -lt 600 ]; do
    OMNI_SOCKET=$SOCK $CLI file_status ${(f)"$(cat $W/added-$1.txt)"} --json --no-launch > $W/status-$1.json 2>/dev/null
    python3 -c "
import json,sys
d=json.load(open('$W/status-$1.json')); fs=d.get('files', d if isinstance(d, list) else [])
ok=[f for f in fs if f.get('indexed') and f.get('up_to_date', True)]
print(len(ok), len(fs)); sys.exit(0 if fs and len(ok)==len(fs) else 1)" > $W/status-$1.txt 2>&1 && return 0
    sleep 3; done; mark "TIMEOUT indexing batch $1: $(cat $W/status-$1.txt)"; }

# 1. ANE from launch: index everything from empty.
wait_log "search and indexing on the Neural Engine" 1200; mark ane-ready
wait_pass 0 1800; mark ane-indexed
mark idle-ane; sleep 20
mark ui-ane; touch $W/cue/1; sleep 45
mark served-ane; load served-ane
mark add-ane; add a 1; wait_indexed a; mark add-ane-done
# 2. To the GPU.
mark switch-gpu; touch $W/cue/2; wait_log "engine reloaded" 300; mark gpu-ready
mark idle-gpu; sleep 20
mark ui-gpu; touch $W/cue/3; sleep 45
mark served-gpu; load served-gpu
mark add-gpu; add g 2; wait_indexed g; mark add-gpu-done
# 3. Back to the ANE.
mark switch-ane; touch $W/cue/4; wait_log "search and indexing on the Neural Engine" 1200 2; mark ane2-ready
mark idle-ane2; sleep 20
mark ui-ane2; touch $W/cue/5; sleep 45
mark served-ane2; load served-ane2
mark add-ane2; add b 3; wait_indexed b; mark add-ane2-done
mark end; touch $W/cue/6; sleep 3
kill $P; sleep 3; kill $PS; rm -f $SOCK
