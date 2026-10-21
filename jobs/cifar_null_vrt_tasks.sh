#!/bin/bash
# cifar_null_vrt_tasks.sh — task file for the VRT Type-I check on CIFAR (cifar_null_vrt.py). 500 repetitions per
# architecture, 100 per GPU task. Every repetition draws a new random network, a new random 50,000 / 10,000 split of the
# 60,000 CIFAR images and a new VRT draw (the Type-I estimate), and also tests the official split for comparison.
# Eval-mode extraction (as Table 1) by default. Options through the environment:
#   ARCHS="r18_c10 r18_c100 r50_c100 simclr_c10 frn_c10"   (also possible: r50_c10 simclr_c100)
#   MODES="eval"        MODES="eval train" adds the Section 4.4 extraction (not for frn_c10, which has no BatchNorm)
#   NREP=500  CHUNK=100
# Usage, on the cluster:  cd /scratch/p318770/ssl_supcon_code_s50 && bash cifar_null_vrt_tasks.sh   (then the sbatch line)
set -e
CODE=${CODE:-/scratch/p318770/ssl_supcon_code_s50}
S50=${S50:-/home3/p318770/venvs/overfitting/benchmark_final/s50}
DATA=${DATA:-/home3/p318770/venvs/overfitting}
OUT=$S50/null_cifar_vrt
ARCHS=${ARCHS:-"r18_c10 r18_c100 r50_c100 simclr_c10 frn_c10"}
MODES=${MODES:-eval}
NREP=${NREP:-500}; CHUNK=${CHUNK:-100}
cd "$CODE"
for f in cifar_null_vrt.py harvest_cifar_null_vrt.py vrt_low_lib.py extract_feats.py train_simclr_supcon.py supcon_fill_vrt.py job_s50_gpu.sh; do
  [ -f "$f" ] || { echo "ERROR: $f missing in $CODE"; exit 1; }
done
for f in "$DATA/cifar-10-batches-py/test_batch" "$DATA/cifar-100-python/test"; do
  [ -f "$f" ] || { echo "ERROR: $f not found (CIFAR files of the paper's extraction)"; exit 1; }
done
mkdir -p "$OUT" s50_logs
F=cifar_null_vrt_tasks_$(date +%Y%m%d_%H%M).txt; : > "$F"
for LO in $(seq 0 "$CHUNK" $((NREP - 1))); do          # chunk by chunk, so the first tasks cover every architecture
  HI=$((LO + CHUNK - 1)); [ $HI -ge "$NREP" ] && HI=$((NREP - 1))
  for A in $ARCHS; do for M in $MODES; do
    [ "$A" = frn_c10 ] && [ "$M" = train ] && continue
    echo "python cifar_null_vrt.py --arch $A --bn_mode $M --reps $LO-$HI --out_dir $OUT" >> "$F"
  done; done
done
N=$(wc -l < "$F")
echo "$N tasks -> $F   (architectures: $ARCHS; modes: $MODES; $NREP repetitions, $CHUNK per task)"
echo "submit:   sbatch --array=0-$((N - 1)) job_s50_gpu.sh $F"
echo "progress: grep -h '^rep ' s50_logs/gpu_<jobid>_*.out | wc -l     (finished repetitions, all tasks)"
echo "harvest:  python harvest_cifar_null_vrt.py $OUT"
