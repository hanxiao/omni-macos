#!/bin/zsh
# Run UITests/ComputeUnitUITests one test at a time - the Neural Engine switch through the real UI -
# with the test app's own GPU time every second, a 20 GB footprint watchdog, and its crash reports.
#
#   OMNI_CHAOS_APP=<copy.app> ./Scripts/compute-unit-ui.sh <out dir> <corpus> <model dir> <warm ane cache> [tests...]
#
# Inside an automation window (Scripts/automation-window.sh), on a copy with its own bundle id
# (Scripts/chaos-app.sh). The model and the corpus must be on the internal disk: the copy has no grant
# for removable volumes, and a model on one put up a TCC prompt the run sat behind until it failed.
# Tools/ane/modestress/uisteps.py <out dir> prints each step with the GPU time it used.
set -u
cd "$(dirname "$0")/.."
OUT=$1; CORPUS=$2; MODEL=$3; CACHE=$4; shift 4
APP=${OMNI_CHAOS_APP:?a renamed copy (Scripts/chaos-app.sh)}
TESTS=($@)
[ ${#TESTS} -gt 0 ] || TESTS=(testSwitchBothWaysFromSettings testRapidToggles testSwitchWhileIndexing testRelaunchInEachMode
                              testChoosesTheNeuralEngineAtFirstRun testTurnedOffDuringTheOneTimeBuild)
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode-26.6.0.app/Contents/Developer}
DD=.build/xcode-rel
ART="$PWD/$DD/SourcePackages/artifacts/swift-tokenizers/TokenizersRust/TokenizersRust.artifactbundle"
apppid() { pgrep -f "$APP/Contents/MacOS/" | head -1; }
mkdir -p $OUT; START=$(date +%s)
for t in $TESTS; do
  base=$OUT/$t; SCRATCH=$OUT/scratch; rm -rf $SCRATCH; mkdir -p $SCRATCH
  ( while true; do
      p=$(apppid)
      if [ -n "$p" ]; then
        echo "$p $(python3 Tools/ane/modestress/gpuq.py $p)" >> $base.gpu
        fp=$(footprint -p $p 2>/dev/null | awk '/phys_footprint:/{v=$2; u=$3; if (u=="GB") v*=1024; if (u=="KB") v/=1024; print int(v)}')
        echo "$(date +%s) $p ${fp:-0}" >> $base.mem
        [ "${fp:-0}" -gt 20480 ] && { echo "WATCHDOG $p ${fp} MB" >> $base.mem; kill -9 $p; }
      fi
      sleep 1
    done ) &
  SAMPLER=$!
  TEST_RUNNER_OMNI_CHAOS_APP=$APP TEST_RUNNER_OMNI_UNIT_CORPUS=$CORPUS TEST_RUNNER_OMNI_UNIT_SCRATCH=$SCRATCH \
  TEST_RUNNER_OMNI_UNIT_ANE_CACHE=$CACHE TEST_RUNNER_OMNI_UNIT_MODEL=$MODEL \
  xcodebuild test-without-building -project Omni.xcodeproj -scheme OmniUITests -configuration Debug \
    -destination 'platform=macOS' -derivedDataPath "$DD" -only-testing:"OmniUITests/ComputeUnitUITests/$t" \
    OTHER_SWIFT_FLAGS="\$(inherited) -Xcc -fmodule-map-file=$ART/include/module.modulemap -Xcc -I$ART/include" \
    OTHER_LDFLAGS="\$(inherited) $ART/apple-macos/libtokenizers_rust.a" > $base.log 2>&1
  rc=$?
  kill $SAMPLER 2>/dev/null; wait $SAMPLER 2>/dev/null
  p=$(apppid); [ -n "$p" ] && kill $p
  for f in $SCRATCH/stderr-*.log(N); do cp $f $base.$(basename $f); done   # the app's perf log, per launch
  # xcodebuild exits 0 when -only-testing matched nothing: say so.
  ran=$(grep -E 'Executed 1 test' $base.log | tail -1); [ -n "$ran" ] || ran="NO TEST RAN"
  echo "$(date +%T) $t rc=$rc $ran" | tee -a $OUT/summary.txt
  # The cold build's compile outlives its client: let it finish before the next test uses the ANE.
  [ $t = testTurnedOffDuringTheOneTimeBuild ] && sleep 120
done
find ~/Library/Logs/DiagnosticReports -newermt "@$START" -name "$(basename $APP .app)*" -exec cp {} $OUT/ \; 2>/dev/null
