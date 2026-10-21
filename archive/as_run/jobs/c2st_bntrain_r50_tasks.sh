#!/bin/bash
# c2st_bntrain_r50_tasks.sh — C2ST on the train-mode BatchNorm features of ResNet-50 (CIFAR-100), the one setting with
# train-mode features where C2ST detects in eval mode (Table 1 onset 121), so that Table 4 gets a C2ST column entry.
# Unified protocol, the s50 code path (base_fill.py, c2st_test): 10,000 points per split, seeds 42-91, every saved epoch.
# Epochs 100 to 200 are listed first (around the eval-mode onset), then 0 to 99 (so an earlier train-mode onset is not
# missed). 5 epochs per CPU task. Results in s50/bntrain_c2st, read by harvest_bn_c2st.py next to the eval-mode results.
# Usage, on the cluster:  cd /scratch/p318770/ssl_supcon_code_s50 && bash c2st_bntrain_r50_tasks.sh   (then the sbatch line)
set -e
CODE=${CODE:-/scratch/p318770/ssl_supcon_code_s50}
FEATS=${FEATS:-/scratch/p318770/model_zoos/ladder_feats}
S50=${S50:-/home3/p318770/venvs/overfitting/benchmark_final/s50}
OUT=$S50/bntrain_c2st
TAG=r50_bntrain
cd "$CODE"
for f in base_fill.py baselines2_stats.py harvest_bn_c2st.py job_s50_cpu.sh; do
  [ -f "$f" ] || { echo "ERROR: $f missing in $CODE"; exit 1; }
done
grep -q "out_suffix" base_fill.py || { echo "ERROR: base_fill.py is not the s50 version (no --out_suffix)"; exit 1; }
n=$(ls "$FEATS/$TAG"/ep*.npz 2>/dev/null | wc -l)
[ "$n" -gt 0 ] || { echo "ERROR: no train-mode features in $FEATS/$TAG"; exit 1; }
echo "$TAG: $n feature files in $FEATS/$TAG"
mkdir -p "$OUT" s50_logs
F=c2st_bntrain_r50_tasks_$(date +%Y%m%d_%H%M).txt; : > "$F"
for LO in $(seq 100 5 200) $(seq 0 5 95); do
  HI=$((LO + 4))
  echo "python base_fill.py --stat c2st --tag $TAG --npz_fmt '$FEATS/$TAG/ep{ep:03d}.npz' --budgets 10000 --every 1 --seeds 42-91 --epochs $LO-$HI --out_dir $OUT --out_suffix _c$LO-$HI" >> "$F"
done
N=$(wc -l < "$F")
echo "$N tasks -> $F   (5 epochs x 50 seeds each, epochs 100-204 first, then 0-99)"
echo "submit:   JB=\$(sbatch --parsable --time=04:00:00 --array=0-$((N - 1)) job_s50_cpu.sh $F); echo \$JB"
echo "harvest:  python harvest_bn_c2st.py"
