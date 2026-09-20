#!/bin/bash
# DRIVE THE REAL APP WITHOUT XCUITEST.
#
#   ./Scripts/chaos-drive.sh <index-dir> [seconds]
#
# The XCUITest runner needs com.apple.dt.AutomationModeUI, which can wedge and which SIP will not
# let a user restart. This needs none of it: the app is launched directly (so the feature flags in
# the environment reach it), and cliclick drives the pointer and the keyboard.
#
# INCOMPLETE, AND HERE FOR WHAT IT ALREADY ESTABLISHES. Launched from a shell the app comes up and
# stays up, but presents NO WINDOW: `System Events` reports 0 windows after two minutes, the
# Window menu offers nothing that opens one, `AXExtrasMenuBar` is missing value, and activating
# the process by id does not summon one either. `-omni.query` applies a query to the model but
# does not present the window. So the remaining piece is whatever XCUIApplication().launch() does
# that a direct exec does not - most likely the activation that makes the MenuBarExtra realise its
# window scene.
#
# Everything downstream of that - the interaction loop, the bounds maths, the log grep - is
# written and unexercised. Whoever picks this up needs only a way to get the window on screen.
#
# Typed TEXT reaches the app; synthetic key PRESSES do not (see CLAUDE.md), so cancels are done the
# way a person actually does them - select-all by dragging, retype over it, click away mid-query.
set -u
SRC=${1:?index dir}; SECS=${2:-200}
APP=/Applications/Omni.app/Contents/MacOS/Omni
W="$(dirname "$SRC")/omni-drive-live"
LOG=${OMNI_DRIVE_LOG:-/tmp/omni-drive.log}

pkill -x Omni 2>/dev/null; sleep 2
rm -rf "$W"; mkdir -p "$W"
for f in "$SRC"/*; do cp -c "$f" "$W/"; done
: > "$LOG"
echo "[drive] index $(ls -la "$W/index.sqlite" | awk '{printf "%.2f GB", $5/1073741824}') at $W"

OMNI_FREE_LIST=${OMNI_FREE_LIST:-} \
  "$APP" -omni.dbDir "$W" -omni.stderrFile "$LOG" -omni.ephemeralUIState YES \
         -omni.serving.enabled NO -omni.query "invoice" >/dev/null 2>&1 &
APPPID=$!
# `-omni.query` is the seam that OPENS THE WINDOW. Omni lives in the menu bar and has no window
# until something asks for one, so without this there is nothing for System Events to measure and
# nothing for the pointer to hit. It also waits for the engine to be ready before searching, which
# is what makes the first interaction meaningful rather than a race.
for _ in $(seq 1 40); do
  osascript -e 'tell application "System Events" to tell process "Omni" to get count of windows' \
    2>/dev/null | grep -qE "^[1-9]" && break
  sleep 2
done
kill -0 $APPPID 2>/dev/null || { echo "[drive] app did not start"; exit 1; }

# Window bounds via System Events, which does NOT activate Omni by name (that would start a
# second instance - see CLAUDE.md).
read -r X Y WI HE < <(osascript -e 'tell application "System Events" to tell process "Omni"
  set {x, y} to position of window 1
  set {w, h} to size of window 1
  return (x as string) & " " & (y as string) & " " & (w as string) & " " & (h as string)
end tell' 2>/dev/null | tr -d ',')
[ -z "${X:-}" ] && { echo "[drive] no window"; kill $APPPID; exit 1; }
echo "[drive] window ${WI}x${HE} at ${X},${Y}"

SEARCH_X=$(( X + WI - 170 )); SEARCH_Y=$(( Y + 26 ))
RESULT_X=$(( X + WI / 3 ));   RESULT_Y=$(( Y + 140 ))
SIDEBAR_X=$(( X + 90 ));      SIDEBAR_Y=$(( Y + 160 ))

QUERIES=("invoice" "porsche" "quarterly revenue" "screenshot" "contract" "tomatoes" "memory budget")
END=$(( $(date +%s) + SECS )); N=0
while [ "$(date +%s)" -lt "$END" ]; do
  Q=${QUERIES[$(( RANDOM % ${#QUERIES[@]} ))]}
  case $(( N % 6 )) in
    0) cliclick "c:${SEARCH_X},${SEARCH_Y}" "t:${Q}" >/dev/null; sleep 1 ;;
    1) # cancel storm: retype three times without letting any land
       cliclick "c:${SEARCH_X},${SEARCH_Y}" >/dev/null
       for _ in 1 2 3; do cliclick "t:${Q}" >/dev/null; sleep 0.3; done ;;
    2) cliclick "c:${RESULT_X},${RESULT_Y}" >/dev/null; sleep 0.6 ;;
    3) # context menu on a result, then dismiss by clicking away
       cliclick "rc:${RESULT_X},${RESULT_Y}" >/dev/null; sleep 0.8
       cliclick "c:$(( X + WI / 2 )),$(( Y + HE - 40 ))" >/dev/null ;;
    4) cliclick "c:${SIDEBAR_X},${SIDEBAR_Y}" >/dev/null; sleep 0.5 ;;
    5) # a drag across the results - the marquee, and a scroll
       cliclick "dd:${RESULT_X},${RESULT_Y}" "dm:$(( RESULT_X + 220 )),$(( RESULT_Y + 180 ))" "du:$(( RESULT_X + 220 )),$(( RESULT_Y + 180 ))" >/dev/null
       cliclick "w:$(( RESULT_X )),$(( RESULT_Y ))" >/dev/null 2>&1 ;;
  esac
  N=$(( N + 1 ))
  kill -0 $APPPID 2>/dev/null || { echo "[drive] APP DIED after $N interactions"; exit 1; }
done
echo "[drive] $N interactions, app alive"
sleep 3
kill $APPPID 2>/dev/null; wait $APPPID 2>/dev/null
echo "[drive] app log, non-system lines:"
grep -vE "task name port|Greymatter|DetachedSignatures|logging-persist|ViewBridge|Performance Diagnostics" "$LOG" | head -20
echo "[drive] index left at $W"
