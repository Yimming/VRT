#!/bin/bash
#SBATCH --job-name=vrt_rev
#SBATCH --array=0-33
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=vrt_rev_%A_%a.out
# Reversed VRT (test-as-reference) at both matched budgets, all 17 settings.
# task i -> TAG = TAGS[i/2], BUDGET = large (even i) / small (odd i)

source /home3/p318770/venvs/simclr_vrt_gpu/venv311/bin/activate
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1

FEATS=/scratch/p318770/model_zoos/ladder_feats     # feature root (adjust if different)
CODE=/scratch/p318770/ssl_supcon_code               # where vrt_reversed_budget.py lives
OUTDIR=/home3/p318770/venvs/overfitting/benchmark_final/reversed
mkdir -p $OUTDIR

TAGS=(r50 v7 ablA ablB ablC ablD wp \
      zoo_c10_m0 zoo_c10_m1 zoo_c10_m2 zoo_c10_m3 zoo_c10_m4 \
      zoo_c100_m0 zoo_c100_m1 zoo_c100_m2 zoo_c100_m3 zoo_c100_m4)
TAG=${TAGS[$((SLURM_ARRAY_TASK_ID / 2))]}
if (( SLURM_ARRAY_TASK_ID % 2 == 0 )); then BUDGET=large; else BUDGET=small; fi

# ---- feature layout confirmed 2026-09-26: $FEATS/<tag>/ep{ep:03d}.npz (both splits in one npz) ----
NPZ="$FEATS/$TAG/ep{ep:03d}.npz"
# If the script reports it cannot identify the train/test keys, add:  --tr_key <k1> --te_key <k2>
KEYS=""

cd $CODE
python vrt_reversed_budget.py --tag $TAG --budget $BUDGET \
  --npz_fmt "$NPZ" $KEYS --epochs 0-1000 \
  --seeds 42-51 --n_perm 2000 \
  --out $OUTDIR/vrt_rev_${TAG}_${BUDGET}.csv
