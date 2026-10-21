#!/bin/bash
#SBATCH --job-name=knn_v2
#SBATCH --array=0-33
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=knn_v2_%A_%a.out
# Schilling / Henze rerun with corrected code (conditional variance, index-based self exclusion).
# 2 budgets x 17 settings. task i: budget = BUDGETS[i / 17], tag = TAGS[i % 17].

source /home3/p318770/venvs/simclr_vrt_gpu/venv311/bin/activate
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1

FEATS=/scratch/p318770/model_zoos/ladder_feats
CODE=/scratch/p318770/ssl_supcon_code
OUTDIR=/home3/p318770/venvs/overfitting/benchmark_final/knn_v2
mkdir -p $OUTDIR

TAGS=(r50 v7 ablA ablB ablC ablD wp \
      zoo_c10_m0 zoo_c10_m1 zoo_c10_m2 zoo_c10_m3 zoo_c10_m4 \
      zoo_c100_m0 zoo_c100_m1 zoo_c100_m2 zoo_c100_m3 zoo_c100_m4)
BUDGETS=(1500 10000)
TAG=${TAGS[$((SLURM_ARRAY_TASK_ID % 17))]}
N=${BUDGETS[$((SLURM_ARRAY_TASK_ID / 17))]}

cd $CODE
python knn_tst_budget.py --tag $TAG --npz_fmt "$FEATS/$TAG/ep{ep:03d}.npz" \
  --n_samples $N --k 3 --n_perm 2000 --seeds 42-51 --epochs 0-1000 --backend torch \
  --out $OUTDIR/knn_${TAG}_n${N}_part${SLURM_ARRAY_TASK_ID}.csv
