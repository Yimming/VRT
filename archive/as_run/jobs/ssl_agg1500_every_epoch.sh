#!/bin/bash
# ssl_agg1500_every_epoch.sh — task file that brings SimCLR MMDAgg at 1,500 points per split from every 5th epoch to every
# epoch (1 to 1000), 50 seeds, both datasets. Epochs already on the every-5th grid are skipped by the resume of
# ssl_s50_fill.py (it reads every ssl_<dataset>_slow_c*.csv file in the output folder; needs the patched version).
# One task = 10 consecutive epochs (8 new), about 4 to 5 h on 8 CPU cores (CPU feature extraction about 10 min per
# checkpoint plus 50 MMDAgg tests). The task file lists the onset windows first, CIFAR-10 691 to 760 and CIFAR-100 581 to 670
# (from just after the last non-rejecting grid epoch to where the grid stays above 90 percent of seeds), then the rest.
# Usage, on the cluster:  cd /scratch/p318770/ssl_supcon_code_s50 && bash ssl_agg1500_every_epoch.sh
# then one of the two sbatch lines it prints (job_s50_cpu.sh runs one line of the task file per array task).
set -e
CODE=${CODE:-/scratch/p318770/ssl_supcon_code_s50}
BF=${BF:-/home3/p318770/venvs/overfitting/benchmark_final/s50}
CK=${CK:-/scratch/p318770/model_zoos/ssl_supcon}
cd "$CODE"
grep -q "file_tag" ssl_s50_fill.py || { echo "ERROR: ssl_s50_fill.py is the old version (no --file_tag); upload the patched one"; exit 1; }
STAMP=$(date +%Y%m%d_%H%M); F=ssl_agg1500_tasks_$STAMP.txt; : > "$F"
line() { echo "python ssl_s50_fill.py --ckpt_dir $1 --dataset $2 --group slow --every 1 --lo $3 --hi $4 --seeds 42-91 --out_dir $BF/ssl --file_tag _e1" >> "$F"; }
declare -A WLO=([cifar10]=691 [cifar100]=581) WHI=([cifar10]=760 [cifar100]=670)
declare -A DIR
for ds in cifar10 cifar100; do
  DIR[$ds]=$(ls -d "$CK"/SimCLR_${ds}_* 2>/dev/null | head -1)
  [ -n "${DIR[$ds]}" ] && [ -d "${DIR[$ds]}" ] || { echo "ERROR: no checkpoint folder SimCLR_${ds}_* in $CK"; exit 1; }
  n=$(ls "${DIR[$ds]}"/ckpt_epoch_*.pth | wc -l); echo "$ds: ${DIR[$ds]} ($n checkpoints)"
done
for ds in cifar10 cifar100; do
  for lo in $(seq ${WLO[$ds]} 10 ${WHI[$ds]}); do line "${DIR[$ds]}" $ds $lo $((lo + 9)); done
done
NW=$(wc -l < "$F")
for ds in cifar10 cifar100; do
  for lo in $(seq 1 10 991); do
    hi=$((lo + 9))
    if [ $hi -ge ${WLO[$ds]} ] && [ $lo -le ${WHI[$ds]} ]; then continue; fi     # window chunk, listed above
    line "${DIR[$ds]}" $ds $lo $hi
  done
done
N=$(wc -l < "$F")
echo "$N tasks -> $F  (the first $NW are the onset windows)"
echo "windows only:  sbatch --array=0-$((NW - 1)) job_s50_cpu.sh $F"
echo "everything:    sbatch --array=0-$((N - 1))%100 job_s50_cpu.sh $F"
