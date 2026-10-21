#!/bin/bash
# cifar_null_base_tasks.sh — task files for the baselines' Type-I check on CIFAR (cifar_null_base.py): the same 500
# networks, random splits and analysis seeds per architecture as the VRT run (cifar_null_vrt.py), 10,000 tier of Table 1.
#   GPU file: MMD, ED, Henze/Schilling on the random and the official split, 100 repetitions per task (about 1 h)
#   CPU file: C2ST and MMDAgg (5,000) on the random split, 10 repetitions per task (about 1 to 1.5 h on 8 cores; only
#             the 20,000 images the two tests draw go through the network, on the CPU)
# Options through the environment:  ARCHS="r18_c10 r18_c100 r50_c100 simclr_c10 frn_c10"  NREP=500  GCHUNK=100  CCHUNK=10
# Usage, on the cluster:  cd /scratch/p318770/ssl_supcon_code_s50 && bash cifar_null_base_tasks.sh   (then the two sbatch lines)
set -e
CODE=${CODE:-/scratch/p318770/ssl_supcon_code_s50}
S50=${S50:-/home3/p318770/venvs/overfitting/benchmark_final/s50}
DATA=${DATA:-/home3/p318770/venvs/overfitting}
OUT=$S50/null_cifar_base
ARCHS=${ARCHS:-"r18_c10 r18_c100 r50_c100 simclr_c10 frn_c10"}
NREP=${NREP:-500}; GCHUNK=${GCHUNK:-100}; CCHUNK=${CCHUNK:-10}
cd "$CODE"
for f in cifar_null_base.py cifar_null_vrt.py harvest_cifar_null.py mmd_tst_fast.py energy_tst_fast.py knn_tst_budget.py \
         baselines2_stats.py base_fill.py extract_feats.py vrt_low_lib.py train_simclr_supcon.py supcon_fill_vrt.py \
         job_s50_gpu.sh job_s50_cpu.sh; do
  [ -f "$f" ] || { echo "ERROR: $f missing in $CODE"; exit 1; }
done
for f in "$DATA/cifar-10-batches-py/test_batch" "$DATA/cifar-100-python/test"; do
  [ -f "$f" ] || { echo "ERROR: $f not found"; exit 1; }
done
mkdir -p "$OUT" s50_logs
STAMP=$(date +%Y%m%d_%H%M)
G=cifar_null_base_gpu_tasks_$STAMP.txt; C=cifar_null_base_cpu_tasks_$STAMP.txt
: > "$G"; : > "$C"
lines() {   # $1 part, $2 chunk, $3 extra arguments, $4 file; chunk by chunk so the first tasks cover every architecture
  for LO in $(seq 0 "$2" $((NREP - 1))); do
    HI=$((LO + $2 - 1)); [ $HI -ge "$NREP" ] && HI=$((NREP - 1))
    for A in $ARCHS; do
      echo "python cifar_null_base.py --part $1 $3--arch $A --reps $LO-$HI --out_dir $OUT" >> "$4"
    done
  done
}
lines gpu "$GCHUNK" "" "$G"
lines cpu "$CCHUNK" "--methods c2st,mmdagg " "$C"
NG=$(wc -l < "$G"); NC=$(wc -l < "$C")
echo "GPU: $NG tasks -> $G   (MMD, ED, Henze/Schilling; random and official split)"
echo "CPU: $NC tasks -> $C   (C2ST, MMDAgg; random split)"
echo "submit:"
echo "  JG=\$(sbatch --parsable --time=02:00:00 --array=0-$((NG - 1)) job_s50_gpu.sh $G); echo \$JG"
echo "  JC=\$(sbatch --parsable --time=04:00:00 --array=0-$((NC - 1))%100 job_s50_cpu.sh $C); echo \$JC"
echo "harvest (all methods, VRT included):  python harvest_cifar_null.py"
