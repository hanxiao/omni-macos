#!/bin/bash
# Run something inside ONE macOS automation-mode authentication.
#
#   ./Scripts/automation-window.sh 40 ./Scripts/migration-chaos.sh
#
# Driving the UI needs automation mode, and enabling it is a password prompt (see
# UITests/AutomationWindowUITests.swift). The mode stays on while any test session holds it, so
# this starts a session that does nothing but stay alive, waits for the state file to appear -
# which is the password having been entered - and only then runs the command. Everything inside
# the window runs unattended.
#
# The prompt appears once, on this script's first few seconds. If nobody answers it, the wait
# below times out and says so rather than leaving a job running against a mode that is off.
#
# SOMEBODY HAS TO BE AT THE KEYBOARD WHEN IT APPEARS. The runner gives up 60 SECONDS after putting
# the prompt up - "Failed to initialize for UI testing: Timed out while enabling automation mode" -
# so waiting longer here cannot help: the holder is already dead by then. Measured by screenshotting
# the screen while it waited, which is the only way to tell "no prompt appeared" from "a prompt
# appeared and nobody was there": the dialog reads "XCTest is trying to Enable UI Automation. Enter
# the password for the user ...".
#
# THERE IS NO TEN-HOUR CAP. A 600-minute hold was read as one; it was only the number passed.
# 1500 minutes holds 1500 minutes - the deadline is this script's argument and nothing else.
#
# TO HOLD THE WINDOW OPEN PAST THE JOB, which is what "do not ask me again today" means, give it a
# command that outlives the work and detach it:
#
#   nohup ./Scripts/automation-window.sh 1500 sleep 86400 >/tmp/omni-auto.log 2>&1 & disown
#
# Every later `xcodebuild test` finds the state file already there and does not re-authenticate.
# `touch /tmp/omni-automation-window.release` ends it early.
set -u
cd "$(dirname "$0")/.."

MINUTES=${1:-30}; shift || true
[ $# -gt 0 ] || { echo "usage: $0 <minutes> <command> [args...]"; exit 2; }

STATE=/var/db/com.apple.dt.automationmode/automation-enabled
RELEASE=/tmp/omni-automation-window.release
rm -f "$RELEASE"
HOLDLOG=/tmp/omni-automation-window.log

# Its own derived data: this xcodebuild runs for the whole window, concurrently with the real one.
OMNI_DD=.build/xcode-hold \
TEST_RUNNER_OMNI_AUTOMATION_HOLD_MINUTES="$MINUTES" \
  ./Scripts/ui-test.sh AutomationWindowUITests >"$HOLDLOG" 2>&1 &
HOLD=$!
trap 'touch "$RELEASE"; wait $HOLD 2>/dev/null' EXIT

echo "waiting for the automation-mode password prompt to be answered (up to 5 min)..."
for _ in $(seq 1 300); do
  [ -e "$STATE" ] && break
  kill -0 $HOLD 2>/dev/null || { echo "holder exited early; see $HOLDLOG"; tail -20 "$HOLDLOG"; exit 1; }
  sleep 1
done
if [ ! -e "$STATE" ]; then
  echo "automation mode never came on - the prompt was not answered. see $HOLDLOG"
  exit 1
fi
echo "automation mode ON, window open for $MINUTES min; running: $*"

"$@"
rc=$?
echo "command finished rc=$rc; releasing the window"
exit $rc
