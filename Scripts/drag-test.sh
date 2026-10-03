#!/bin/zsh
# Run FileDragUITests: real mouse drags out of a renamed copy of the app onto DropProbe.
#
#   ./Scripts/drag-test.sh [Test]      e.g. FileDragUITests/testOmniRefusesItsOwnDrag
#
# Needs an UNLOCKED screen and UI automation mode (the password prompt, or a running
# Scripts/automation-window.sh). Launches a COPY of the build (io.hanxiao.omni.chaos): XCUITest's
# launch() quits whatever app has the target bundle id, which would be the user's own Omni.
# Writes nothing to the user's clipboard that it does not put back (see the Command-C test).
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=.build/drag-test
mkdir -p "$OUT"
./Scripts/build-app.sh Release
./Scripts/chaos-app.sh "$OUT/OmniDragTest.app"
./Scripts/drop-probe.sh "$OUT"
# The corpus, outside every sandbox container (see FileDragUITests.setUp). Distinct documents:
# near-identical ones are stacked as duplicates and leave too few rows to drag.
ROOT=/private/tmp/omni-drag-test
rm -rf "$ROOT"; mkdir -p "$ROOT/corpus" "$ROOT/db"
i=0
for t in "solar panel efficiency in winter" "a sourdough bread recipe with rye" \
         "a marathon training plan for beginners" "the history of the printing press" \
         "tuning a guitar by ear" "migrating birds over the Alps" "a budget for a kitchen remodel" \
         "how tides follow the moon" "caring for a fiddle leaf fig" "the rules of chess openings" \
         "brewing pour-over coffee" "repairing a bicycle chain"; do
  for l in $(seq 0 19); do echo "Notes on $t, part $l: details and observations about $t."; done > "$ROOT/corpus/notes$i.txt"
  i=$((i+1))
done
TEST_RUNNER_OMNI_CHAOS_APP="$PWD/$OUT/OmniDragTest.app" \
TEST_RUNNER_OMNI_DROPPROBE_APP="$PWD/$OUT/DropProbe.app" \
TEST_RUNNER_OMNI_DRAG_ROOT="$ROOT" \
OMNI_UI_CONFIG=Release ./Scripts/ui-test.sh "${1:-FileDragUITests}"
