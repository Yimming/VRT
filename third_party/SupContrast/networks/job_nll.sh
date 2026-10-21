#!/bin/bash
#SBATCH --job-name=nll62
#SBATCH --partition=gpushort
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --output=nll62_%j.out
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source /home3/p318770/venvs/simclr_vrt_gpu/venv311/bin/activate
python eval_nll_gap.py
