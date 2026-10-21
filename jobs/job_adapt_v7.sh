#!/bin/bash
#SBATCH --job-name=adapt_v7
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=adapt_v7_%j.out

module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source /home3/p318770/venvs/simclr_vrt_gpu/venv311/bin/activate

python adaptive_theta.py --model v7 --feats_dir /scratch/p318770/model_zoos/ladder_feats/v7 \
  --out_dir /scratch/p318770/model_zoos/ladder_out/adapt_v7 --epochs 10-45
