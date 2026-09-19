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
# And the pattern has to name the END of the job, not something that appears part way through it.
# `Executed [0-9]+ tests` matched the first sub-suite and reported success 15 s into a 330 s run;
# for a test job the honest signal is the process exiting, not a line in its log.
EXPECT=${1:?expected seconds}; LOG=${2:?logfile}; PAT=${3:?done pattern}; PROC=${4:-omni-verify}
# GRACE, because the watched process often is not the first thing to run: a test job compiles for
# a minute before xctest exists at all, and the first version of this script called that "exited
# after 0s" and gave up instantly. Absence only counts once the process has been seen alive.
GRACE=${5:-240}
START=$(date +%s); SEEN=0
while :; do
  grep -qE "$PAT" "$LOG" 2>/dev/null && { echo "[await] done in $(( $(date +%s) - START ))s"; exit 0; }
  if pgrep -x "$PROC" >/dev/null; then SEEN=1; fi
  if [ "$SEEN" = 1 ] && ! pgrep -x "$PROC" >/dev/null; then
    echo "[await] $PROC exited after $(( $(date +%s) - START ))s without matching"; tail -3 "$LOG"; exit 1
  fi
  if [ "$SEEN" = 0 ] && [ $(( $(date +%s) - START )) -gt "$GRACE" ]; then
    echo "[await] $PROC never started within ${GRACE}s"; tail -3 "$LOG"; exit 1
  fi
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
