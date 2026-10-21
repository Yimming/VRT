#!/bin/bash
# cifar_null_rule_tasks.sh — task file for the blank control of the paper's decision rule (cifar_null_rule.py): the
# paper's VRT procedure (official split, seeds 42 to 91, median p < 0.05) on 30 untrained networks per architecture,
# the CIFAR-100 models of Table 1 and the CIFAR-10 models as comparison. 10 networks per GPU task (about 1 h).
# Options through the environment:
#   ARCHS="r18_c100 r50_c100 simclr_c100 r18_c10 frn_c10 simclr_c10"   NNET=30   CHUNK=10
# Usage, on the cluster:  cd /scratch/p318770/ssl_supcon_code_s50 && bash cifar_null_rule_tasks.sh   (then the sbatch line)
set -e
CODE=${CODE:-/scratch/p318770/ssl_supcon_code_s50}
S50=${S50:-/home3/p318770/venvs/overfitting/benchmark_final/s50}
DATA=${DATA:-/home3/p318770/venvs/overfitting}
OUT=$S50/null_cifar_rule
ARCHS=${ARCHS:-"r18_c100 r50_c100 simclr_c100 r18_c10 frn_c10 simclr_c10"}
NNET=${NNET:-30}; CHUNK=${CHUNK:-10}
cd "$CODE"
for f in cifar_null_rule.py harvest_cifar_null_rule.py cifar_null_vrt.py vrt_low_lib.py extract_feats.py \
         train_simclr_supcon.py supcon_fill_vrt.py job_s50_gpu.sh; do
  [ -f "$f" ] || { echo "ERROR: $f missing in $CODE"; exit 1; }
done
for f in "$DATA/cifar-10-batches-py/test_batch" "$DATA/cifar-100-python/test"; do
  [ -f "$f" ] || { echo "ERROR: $f not found"; exit 1; }
done
mkdir -p "$OUT" s50_logs
F=cifar_null_rule_tasks_$(date +%Y%m%d_%H%M).txt; : > "$F"
for LO in $(seq 0 "$CHUNK" $((NNET - 1))); do
  HI=$((LO + CHUNK - 1)); [ $HI -ge "$NNET" ] && HI=$((NNET - 1))
  for A in $ARCHS; do
    echo "python cifar_null_rule.py --arch $A --nets $LO-$HI --out_dir $OUT" >> "$F"
  done
done
N=$(wc -l < "$F")
echo "$N tasks -> $F   (architectures: $ARCHS; $NNET networks, $CHUNK per task)"
echo "submit:   JR=\$(sbatch --parsable --time=02:00:00 --array=0-$((N - 1)) job_s50_gpu.sh $F); echo \$JR"
echo "harvest:  python harvest_cifar_null_rule.py        (add -v to list every network)"
