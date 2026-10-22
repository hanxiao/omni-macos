#!/bin/zsh
# Does the watcher's queue keep moving? An isolated Omni over a generated 20,000-file corpus, under
# file churn and agent load, with a sentinel file every 20 s; a sentinel still unindexed after
# 240 s is a STALL, reported with what list_sources says.
#
#   Tools/agenteval/drain-soak.sh <app bundle> <new scratch dir> [minutes=20] [port=51398]
#   OMNI_PERF_SCRIPT="repeat:2000;block:cpu:450" OMNI_PERF_SCRIPT_SETTLE=0.35 ... keeps the main
#   thread busy, which is what it took for 0.18.0 to park itself (index.md, "The index that paused
#   itself"). Never against a real index: everything here is generated, under <scratch>.
#
# Churn: an append every 0.3 s (agents' event logs), a new file every 2 s, a deletion every 10 s,
# 2,500 files every 75 s (reconciles longer than the 30 s events wait limit). Load: four clients
# searching back to back. Port 51399 is the eval instance's, 51299 a real install's.
set -u
APP=${1:?app bundle}; S=${2:?new scratch dir}; MIN=${3:-20}; PORT=${4:-51398}
[[ -e $S ]] && { echo "$S exists; give a new scratch dir" >&2; exit 1; }
curl -s -m 1 "http://127.0.0.1:$PORT/health" >/dev/null && { echo "something already serves $PORT" >&2; exit 1; }
R=$S/root; mkdir -p $R/live $S/db
python3 -I - $R/corpus 20000 1 $S/burst 2500 7 <<'EOF'
import os, random, sys
vocab = ("garden invoice harbor lantern meeting budget orchard violin glacier tenant pipeline quarterly "
         "receipt compost migration salmon telescope contract parcel ferry kernel archive mortgage canyon "
         "festival bakery voltage pollen sculpture ledger tram dossier thunder saffron courier atlas").split()
for d, n, seed in ((sys.argv[1], int(sys.argv[2]), int(sys.argv[3])), (sys.argv[4], int(sys.argv[5]), int(sys.argv[6]))):
    rng = random.Random(seed)
    for i in range(n):
        sub = os.path.join(d, f"g{i % 50:02d}"); os.makedirs(sub, exist_ok=True)
        words = [rng.choice(vocab) for _ in range(600)]
        for k in range(12, 600, 13): words[k] += "."
        open(os.path.join(sub, f"doc-{i:06d}.txt"), "w").write(f"Document {i}. " + " ".join(words) + "\n")
EOF
OMNI_PERF_LOG=1 "$APP/Contents/MacOS/Omni" -omni.dbDir $S/db -omni.addedFolders "(\"$R\")" \
  -omni.modelDir "$HOME/Library/Application Support/Omni/nano" -omni.ephemeralUIState YES \
  -omni.serving.enabled YES -omni.serving.port $PORT -omni.socketPath $S/omni.sock \
  -omni.link.bonjour NO -omni.link.port $((PORT + 100)) -omni.link.name drainsoak -omni.clipboard.enabled NO \
  -omni.ocr.cache.dir $S/ocr -omni.profiling.uploadEnabled NO -NSAppSleepDisabled YES > $S/omni.log 2>&1 &
P=$!
( while kill -0 $P 2>/dev/null; do   # an isolated instance left running was once seen at 100 GB
    gb=$(footprint -p $P 2>/dev/null | sed -n 's/.*Footprint: \([0-9]*\)[0-9.]* GB.*/\1/p' | head -1)
    [[ -n $gb ]] && (( gb >= 16 )) && { echo "footprint ${gb} GB: killed" >> $S/omni.log; kill -9 $P; }
    sleep 5
  done ) > /dev/null 2>&1 &
tool() { curl -s -m 30 http://127.0.0.1:$PORT/mcp -H 'Content-Type: application/json' -H 'MCP-Protocol-Version: 2025-11-25' \
  -d "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/call\",\"params\":{\"name\":\"$1\",\"arguments\":$2}}" \
  | python3 -I -c "import json,sys;print(chr(10).join(c.get('text','') for c in json.load(sys.stdin)['result']['content']))" 2>/dev/null; }
for i in {1..360}; do grep -q "launch ready" $S/omni.log 2>/dev/null && break; sleep 0.5; done
T0=$(date +%s); END=$(( T0 + MIN * 60 ))
( n=0; while (( $(date +%s) < END )); do n=$((n+1)); echo "{\"t\":$n,\"text\":\"delta $RANDOM\"}" >> $R/live/events-$((n % 5)).jsonl; sleep 0.3; done ) &
( n=0; while (( $(date +%s) < END )); do n=$((n+1)); echo "New note $n: the heron $RANDOM crossed the canal." > $R/live/new-$n.txt
    (( n > 5 && n % 5 == 0 )) && rm -f $R/live/new-$((n-5)).txt; sleep 2; done ) &
( k=0; sleep 40; while (( $(date +%s) < END )); do k=$((k+1)); cp -c -R $S/burst $R/burst-$k; sleep 75; done ) &
for c in 1 2 3 4; do ( q=(garden invoice harbor lantern "quarterly budget" "salmon migration" "ferry timetable" "okapi ledger")
    while (( $(date +%s) < END )); do tool search "{\"query\":\"${q[$((RANDOM % 8 + 1))]} $RANDOM\",\"top_k\":5}" > /dev/null; done ) & done
typeset -A born; worst=0; stalls=0; s=0; last=0
while (( $(date +%s) < END + 60 )); do
  now=$(date +%s)
  if (( now < END && now - last >= 20 )); then s=$((s+1)); f=$R/sentinel-$s.txt; echo "Sentinel $s: the okapi ledger $RANDOM." > $f; born[$f]=$now; last=$now; fi
  for f in ${(k)born}; do
    if tool file_status "{\"paths\":[\"$f\"]}" | grep -q "up to date"; then
      lat=$(( now - born[$f] )); (( lat > worst )) && worst=$lat; unset "born[$f]"
    elif (( now - born[$f] > 240 )); then
      stalls=$((stalls+1)); echo "[$(( now - T0 ))s] STALL: ${f:t} unindexed for $(( now - born[$f] ))s; $(tool list_sources '{}' | head -1 | cut -c1-160)"; unset "born[$f]"
    fi
  done
  sleep 5
done
echo "sentinels=$s worst_latency=${worst}s stalls=$stalls; passes $(grep -c pass-done $S/omni.log) ($(grep -c 'cancelled=1' $S/omni.log) paused for file events); watchdog lines $(grep -c index-watchdog $S/omni.log)"
kill $P; wait 2>/dev/null
