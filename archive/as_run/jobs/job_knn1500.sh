#!/bin/bash
#SBATCH --job-name=knn1500
#SBATCH --array=0-16
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=knn1500_%A_%a.out
# Schilling / Henze at the small matched budget, n=1,500 per split, all 17 settings.

source /home3/p318770/venvs/simclr_vrt_gpu/venv311/bin/activate
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1

FEATS=/scratch/p318770/model_zoos/ladder_feats
CODE=/scratch/p318770/ssl_supcon_code
OUTDIR=/home3/p318770/venvs/overfitting/benchmark_final/knn1500
mkdir -p $OUTDIR

TAGS=(r50 v7 ablA ablB ablC ablD wp \
      zoo_c10_m0 zoo_c10_m1 zoo_c10_m2 zoo_c10_m3 zoo_c10_m4 \
      zoo_c100_m0 zoo_c100_m1 zoo_c100_m2 zoo_c100_m3 zoo_c100_m4)
TAG=${TAGS[$SLURM_ARRAY_TASK_ID]}

cd $CODE
python knn_tst_budget.py --tag $TAG --npz_fmt "$FEATS/$TAG/ep{ep:03d}.npz" \
  --n_samples 1500 --k 3 --n_perm 2000 --seeds 42-51 --epochs 0-1000 --backend torch \
  --out $OUTDIR/knn_${TAG}_n1500_part${SLURM_ARRAY_TASK_ID}.csv
