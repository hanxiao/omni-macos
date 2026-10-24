#!/bin/zsh
# The Settings benchmark (omni-verify bench, the paper's suite) for one model on one unit, with the
# system's power by unit every second and the bench process's own GPU time every 2 s.
#   benchgoal.sh <out dir> <nano|small> <gpu|ane> <ane cache>
D=$1; MODEL=$2; ARM=$3; CACHE=$4; B=${0:A:h}
N="$HOME/Library/Application Support/Omni/nano"
SM=/Volumes/han2tb/ai-models/huggingface/hub/models--jinaai--jina-embeddings-v5-omni-small-mlx/snapshots/716c5b684db3f6ba574dd9b4f6b14af3b2eb8bda
dir=$N; [ $MODEL = small ] && dir=$SM
O=$D/$MODEL-$ARM; rm -rf $O; mkdir -p $O
${B:h}/powerlog --stream 1 > $O/power.log &
PS=$!
cd ${B:h:h:h}
echo "$(date +%s.%N | cut -c1-14) start" > $O/marks.log
if [ $ARM = ane ]; then
  OMNI_PERF_LOG=1 OMNI_COMPUTE=neuralEngine OMNI_ANE_CACHE=$CACHE .build/release/omni-verify bench "$dir" --out $O/bench.json > $O/bench.log 2>&1 &
else
  OMNI_PERF_LOG=1 .build/release/omni-verify bench "$dir" --out $O/bench.json > $O/bench.log 2>&1 &
fi
P=$!; echo $P > $O/pid
$B/sampler.sh $P $O &
wait $P; echo "exit $?" > $O/exit
echo "$(date +%s.%N | cut -c1-14) end" >> $O/marks.log
sleep 2; kill $PS
