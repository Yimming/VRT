#!/bin/bash
# official_extra_tasks.sh — two CPU task files for the official-split questions of the CIFAR null:
#   1. C2ST and MMDAgg on the OFFICIAL split (cifar_null_base.py --part cpu --splits official): the same 500 networks and
#      seeds per architecture as the random-split run, 10 repetitions per task, own files *_official.csv
#   2. the two controls of cifar_official_check.py: pixels (no network) for CIFAR-10 and CIFAR-100, 100 repetitions,
#      20 per task; random ResNet-18 with shuffled extraction order for CIFAR-10 and CIFAR-100, 100 repetitions, 10 per task
# Usage, on the cluster:  cd /scratch/p318770/ssl_supcon_code_s50 && bash official_extra_tasks.sh   (then the sbatch lines)
set -e
CODE=${CODE:-/scratch/p318770/ssl_supcon_code_s50}
S50=${S50:-/home3/p318770/venvs/overfitting/benchmark_final/s50}
ARCHS=${ARCHS:-"r18_c10 r18_c100 r50_c100 simclr_c10 frn_c10"}
cd "$CODE"
for f in cifar_null_base.py cifar_official_check.py harvest_official_check.py harvest_cifar_null.py cifar_null_vrt.py \
         mmd_tst_fast.py energy_tst_fast.py vrt_low_lib.py baselines2_stats.py base_fill.py job_s50_cpu.sh; do
  [ -f "$f" ] || { echo "ERROR: $f missing in $CODE"; exit 1; }
done
grep -q "_official" cifar_null_base.py || { echo "ERROR: cifar_null_base.py is the old version (no own file for --splits); upload the new one"; exit 1; }
mkdir -p "$S50/null_cifar_base" "$S50/official_check" s50_logs
STAMP=$(date +%Y%m%d_%H%M)
A=official_cpu_tasks_$STAMP.txt; K=official_check_tasks_$STAMP.txt
: > "$A"; : > "$K"
for LO in $(seq 0 10 490); do
  for AR in $ARCHS; do
    echo "python cifar_null_base.py --part cpu --methods c2st,mmdagg --splits official --arch $AR --reps $LO-$((LO + 9)) --out_dir $S50/null_cifar_base" >> "$A"
  done
done
for DS in cifar10 cifar100; do
  for LO in $(seq 0 20 80); do
    echo "python cifar_official_check.py --features pixels --dataset $DS --reps $LO-$((LO + 19)) --out_dir $S50/official_check" >> "$K"
  done
  for LO in $(seq 0 10 90); do
    echo "python cifar_official_check.py --features r18_shuffled --dataset $DS --reps $LO-$((LO + 9)) --out_dir $S50/official_check" >> "$K"
  done
done
NA=$(wc -l < "$A"); NK=$(wc -l < "$K")
echo "1. C2ST + MMDAgg, official split: $NA tasks -> $A"
echo "2. controls (pixels, shuffled extraction): $NK tasks -> $K"
echo "submit:"
echo "  JO=\$(sbatch --parsable --time=04:00:00 --array=0-$((NA - 1))%100 job_s50_cpu.sh $A); echo \$JO"
echo "  JK=\$(sbatch --parsable --time=04:00:00 --array=0-$((NK - 1)) job_s50_cpu.sh $K); echo \$JK"
echo "harvest:  python harvest_cifar_null.py        (C2ST and MMDAgg now also in the OFFICIAL SPLIT table)"
echo "          python harvest_official_check.py    (the two controls)"
