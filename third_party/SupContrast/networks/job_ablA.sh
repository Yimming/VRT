#!/bin/bash
#SBATCH --job-name=ladder_ablA
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=ladder_ablA_%j.out

module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source /home3/p318770/venvs/simclr_vrt_gpu/venv311/bin/activate

FEATS=/scratch/p318770/model_zoos/ladder_feats
OUT=/scratch/p318770/model_zoos/ladder_out

python ladder_stats.py --model ablA --feats_dir $FEATS/ablA --out_dir $OUT/ladder_ablA_par
