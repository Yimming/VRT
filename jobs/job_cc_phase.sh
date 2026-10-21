#!/bin/bash
#SBATCH --job-name=cc_phase
#SBATCH --partition=regular
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/p318770/ssl_supcon_code/cc_phase_%j.out
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source /home3/p318770/venvs/simclr_vrt_gpu/venv311/bin/activate
cd /scratch/p318770/ssl_supcon_code
F=/scratch/p318770/model_zoos/ladder_feats
O=/scratch/p318770/model_zoos/ladder_out
for M in phase_bs8 phase_bs128 phase_bs1024; do
  for EP in 100 125 150; do
    python cc_tests.py --model $M --dataset cifar100 --feats_dir $F/$M \
        --out_dir $O --epochs $EP --n_perm 9999
  done
done
