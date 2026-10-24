#!/bin/zsh
# Every 2 s while <pid> lives: per-queue GPU ns -> gpu.log, footprint MB -> fp.log. Kills it past 20 GB.
P=$1; D=$2
while kill -0 $P 2>/dev/null; do
  python3 $(dirname $0)/gpuq.py $P >> $D/gpu.log
  f=$(footprint -p $P 2>/dev/null | awk '/phys_footprint:/{v=$2; if ($3=="GB") v*=1024; if ($3=="KB") v/=1024; printf "%d", v}')
  [ -n "$f" ] && echo "$(date +%s) $f" >> $D/fp.log
  [ -n "$f" ] && [ "$f" -gt 20480 ] && { kill $P; echo "$(date +%s) KILLED at $f MB" >> $D/fp.log; }
  sleep 2
done
