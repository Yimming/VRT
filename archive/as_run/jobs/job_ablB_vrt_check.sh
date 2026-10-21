#!/bin/bash
#SBATCH --job-name=ablB_vrt
#SBATCH --array=0-1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=05:00:00
#SBATCH --output=ablB_vrt_%A_%a.out
# Configuration B (10x weight decay): forward VRT at 10,000 per split on the eval-mode features (not yet in the
# vrt_fwd pipeline) and on the train-mode features, each with 50 seeds (42-91), to check the train-mode onset 47.
# Task 0 eval-mode features, task 1 train-mode features. Existing (epoch, seed) rows are kept (resume).
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source /home3/p318770/venvs/simclr_vrt_gpu/venv311/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
F=/scratch/p318770/model_zoos/ladder_feats
BF=/home3/p318770/venvs/overfitting/benchmark_final
cd /scratch/p318770/ssl_supcon_code
if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
  python vrt_forward_budget.py --tag ablB --budget large --npz_fmt "$F/ablB/ep{ep:03d}.npz" --epochs 0-50 --seeds 42-91 --n_perm 2000 --out $BF/vrt_fwd/vrt_fwd_ablB_large.csv
else
  python vrt_forward_budget.py --tag ablB_bntrain --budget large --npz_fmt "$F/ablB_bntrain/ep{ep:03d}.npz" --epochs 0-50 --seeds 42-91 --n_perm 2000 --out $BF/bntrain/vrt_fwd/vrt_fwd_ablB_bntrain_large.csv
fi
echo DONE
