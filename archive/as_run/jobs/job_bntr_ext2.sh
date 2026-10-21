#!/bin/bash
#SBATCH --job-name=bnext2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=05:00:00
#SBATCH --output=/scratch/p318770/ssl_supcon_code/bnext2_%j.out
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source /home3/p318770/venvs/simclr_vrt_gpu/venv311/bin/activate
cd /scratch/p318770/ssl_supcon_code
python extract_bntrain2.py
