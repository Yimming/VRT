#!/bin/bash
#SBATCH --job-name=ksweep15x
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=ksweep15x_%j.out
# Small-budget VRT (k=15, rebuttal script ksweep15_fill.py, unchanged) on the settings it did not cover:
# Phase Transition Zoo (3 checkpoints each) and the two ImageNet DeepResNet50 schemes (9 checkpoints each).
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source /home3/p318770/venvs/simclr_vrt_gpu/venv311/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
FEATS=/scratch/p318770/model_zoos/ladder_feats
OUT=/home3/p318770/venvs/overfitting/benchmark_final/ksweep15_extra
mkdir -p $OUT
cd /scratch/p318770/ssl_supcon_code
for U in phase_bs8 phase_bs128 phase_bs1024 imgnet_simple imgnet_adv; do
  python ksweep15_fill.py --unit $U --feats_dir $FEATS/$U --out_dir $OUT --lo 0 --hi 1000
done
echo ALL DONE
