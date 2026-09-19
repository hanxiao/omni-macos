#!/bin/bash
# Wait for a job, but only for as long as it is supposed to take.
#
#   ./Scripts/await-job.sh <expected-seconds> <logfile> <done-pattern> [process-name]
#
# A plain `until grep DONE log; do sleep; done` waits forever and cannot tell "still working" from
# "wedged". Measured cost of that: a splitdry that normally runs 66 s sat for 24 MINUTES on a query
# whose join had stopped using its index, with the wait loop reporting nothing the whole time,
# because the phases before it had printed and the phase that was stuck never would.
#
# So: past the expected time this stops waiting and starts diagnosing - it samples the process and
# prints where it actually is. An overrun is a signal, not a reason to keep sleeping.
EXPECT=${1:?expected seconds}; LOG=${2:?logfile}; PAT=${3:?done pattern}; PROC=${4:-omni-verify}
START=$(date +%s)
while :; do
  grep -qE "$PAT" "$LOG" 2>/dev/null && { echo "[await] done in $(( $(date +%s) - START ))s"; exit 0; }
  pgrep -x "$PROC" >/dev/null || { echo "[await] $PROC exited after $(( $(date +%s) - START ))s without matching"; tail -3 "$LOG"; exit 1; }
  ELAPSED=$(( $(date +%s) - START ))
  if [ "$ELAPSED" -gt "$EXPECT" ]; then
    echo "[await] OVERRUN: ${ELAPSED}s against an expected ${EXPECT}s - sampling instead of waiting"
    P=$(pgrep -x "$PROC" | head -1)
    sample "$P" 3 -f /tmp/await-sample.txt >/dev/null 2>&1
    echo "[await] deepest frames:"; grep -E "^ +[0-9]+ " /tmp/await-sample.txt | tail -6 | cut -c1-110
    echo "[await] last log lines:"; tail -3 "$LOG"
    exit 2
  fi
  sleep 15
done
