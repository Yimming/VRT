#!/bin/bash
#SBATCH --job-name=cc_v7
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-1
#SBATCH --output=cc_v7_%A_%a.out

module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source /home3/p318770/venvs/simclr_vrt_gpu/venv311/bin/activate

FEATS=/scratch/p318770/model_zoos/ladder_feats
OUT=/home3/p318770/venvs/overfitting/benchmark_final

case $SLURM_ARRAY_TASK_ID in
  0) python cc_tests.py --model v7 --dataset cifar10 --feats_dir $FEATS/v7 --out_dir $OUT --epochs 199 --n_perm 9999 ;;
  1) python cc_tests.py --model v7_traj --dataset cifar10 --feats_dir $FEATS/v7 --out_dir $OUT --n_perm 999 ;;
esac
