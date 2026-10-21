#!/bin/bash
#SBATCH --job-name=knn10k
#SBATCH --array=0-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=knn10k_%A_%a.out
# Schilling / Henze at n=10,000 per split (paper Appendix B used 2,000), k=3, 10 seeds, 2,000 perms.
# torch backend (GPU kNN graph, neighbours identical to sklearn). Full per-epoch grid.
# task 0 = r50 (epochs 0-200), task 1 = v7 (epochs 0-199)

source /home3/p318770/venvs/simclr_vrt_gpu/venv311/bin/activate
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1

FEATS=/scratch/p318770/model_zoos/ladder_feats
CODE=/scratch/p318770/ssl_supcon_code
OUTDIR=/home3/p318770/venvs/overfitting/benchmark_final/knn10k
mkdir -p $OUTDIR

TAGS=(r50 v7)
TAG=${TAGS[$SLURM_ARRAY_TASK_ID]}

cd $CODE
python knn_tst_budget.py --tag $TAG --npz_fmt "$FEATS/$TAG/ep{ep:03d}.npz" \
  --n_samples 10000 --k 3 --n_perm 2000 --seeds 42-51 --epochs 0-300 --backend torch \
  --out $OUTDIR/knn_${TAG}_n10000_part${SLURM_ARRAY_TASK_ID}.csv
