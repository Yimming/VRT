#!/bin/bash
# Lists every (dataset, epoch) of the SimCLR MMDAgg 5,000 grid evaluated so far (all sslagg_*.csv rows at b5000)
# into ssl_agg_tasks.txt, one "dataset epoch" per line, and prints the sbatch command for the 50-seed completion.
D=/home3/p318770/venvs/overfitting/benchmark_final/ssl_baselines
OUT=ssl_agg_tasks.txt
cat $D/sslagg_*.csv 2>/dev/null | awk -F, '$3=="b5000" && $4=="mmdagg" {print $1, $2}' | sort -u -k1,1 -k2,2n > $OUT
N=$(wc -l < $OUT)
for ds in cifar10 cifar100; do echo "$ds: $(awk -v d=$ds '$1==d' $OUT | wc -l) epochs: $(awk -v d=$ds '$1==d {printf "%s ", $2}' $OUT)"; done
echo "$N tasks -> $OUT"
echo "submit:  sbatch --array=0-$((N-1)) job_ssl_mmdagg5k_seeds.sh $OUT"
