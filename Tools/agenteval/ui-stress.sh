#!/bin/zsh
# UI stress WITHOUT automation mode (which needs a password at the keyboard): an isolated instance
# runs OMNI_PERF_SCRIPT rounds - typing, views, selection, Find Similar, browsing, sidebar, Recents,
# sort, the intent search - with hangwatch and the footprint watchdog, and reports stalls and
# memory over time.   Tools/agenteval/ui-stress.sh <app> <new out prefix> [rounds]  (BED=testbed)
APP=${1:?app bundle}; OUT=${2:?out prefix}; ROUNDS=${3:-25}
BED=${BED:-/Volumes/han2tb/omni-eval}; T=$BED/tree
ROUND="type:how much is the security deposit;view:grid;view:list;select:0;key:down;key:down;key:down;similar:$T/Documents/Home/Lease agreement 2026.docx;sidebar;sidebar;browse:$T/Documents;view:grid;view:list;recents;clear;type:boat engine repair;sort:date;sort:relevance;select:1;key:down;clear;search:Kaution Wohnung;intent:photos of animals;clear"
SCRIPT="repeat:1;wait:1"
for i in $(seq 1 $ROUNDS); do SCRIPT="$SCRIPT;$ROUND"; done
rm -rf $OUT.dir
OMNI_PERF_LOG=1 OMNI_PERF_SCRIPT_SETTLE=0.6 OMNI_PERF_SCRIPT="$SCRIPT" EXTRA_ARGS="-omni.hangwatch YES -omni.hangwatchMs 250 -omni.hangwatchFile $OUT.hang" \
  $(dirname "$0")/serve-eval.sh $BED $OUT.dir $APP > /dev/null
P=$(cat $OUT.dir/pid)
for i in {1..1800}; do grep -q "script done" $OUT.dir/omni.log 2>/dev/null && break; kill -0 $P 2>/dev/null || break; sleep 1; done
alive=$(kill -0 $P 2>/dev/null && echo yes || echo NO)
kill $P 2>/dev/null; sleep 2
python3 - $OUT.dir/footprint.log $OUT.hang $OUT.dir/omni.log <<'PY'
import sys
fp=[tuple(map(int,l.split())) for l in open(sys.argv[1]) if l.strip()]
t0=fp[0][0]
mb=[(t-t0, kb/1024) for t,kb in fp]
n=len(mb); q=n//4
early=sum(v for _,v in mb[q:2*q])/max(1,q); late=sum(v for _,v in mb[3*q:])/max(1,n-3*q)
try: hangs=[l for l in open(sys.argv[2]).read().splitlines() if l.strip()]
except FileNotFoundError: hangs=[]
steps=sum(1 for l in open(sys.argv[3]) if "] script " in l)
print(f"steps={steps} duration={mb[-1][0]}s footprint start={mb[0][1]:.0f}MB peak={max(v for _,v in mb):.0f}MB 2nd-quarter-avg={early:.0f}MB last-quarter-avg={late:.0f}MB stalls>250ms={len(hangs)}")
PY
echo "alive at end: $alive"
