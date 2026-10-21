#!/bin/bash
#SBATCH --job-name=phase_fill
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=phase_fill_%j.out
# Phase Transition Zoo (bs8 / bs128 / bs1024, checkpoints 100, 125, 150): the pieces the rebuttal's 17-setting
# matched-budget runs did not cover. Reversed VRT at both budgets and Schilling / Henze at both budgets,
# 10 seeds, 2,000 permutations. Outputs go to the same directories as the other settings.
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source /home3/p318770/venvs/simclr_vrt_gpu/venv311/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
FEATS=/scratch/p318770/model_zoos/ladder_feats
BF=/home3/p318770/venvs/overfitting/benchmark_final
cd /scratch/p318770/ssl_supcon_code
for TAG in phase_bs8 phase_bs128 phase_bs1024; do
  NPZ="$FEATS/$TAG/ep{ep:03d}.npz"
  for B in large small; do
    python vrt_reversed_budget.py --tag $TAG --budget $B --npz_fmt "$NPZ" --epochs 0-1000 --seeds 42-51 --n_perm 2000 \
      --out $BF/reversed/vrt_rev_${TAG}_${B}.csv
  done
  for N in 1500 10000; do
    python knn_tst_budget.py --tag $TAG --npz_fmt "$NPZ" --n_samples $N --k 3 --n_perm 2000 --seeds 42-51 --epochs 0-1000 \
      --backend torch --out $BF/knn_v2/knn_${TAG}_n${N}_part${TAG}.csv
  done
done
echo ALL DONE
