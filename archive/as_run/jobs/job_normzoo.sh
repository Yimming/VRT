#!/bin/bash
#SBATCH --job-name=normzoo
#SBATCH --partition=regular
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/p318770/ssl_supcon_code/normzoo_%j.out
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source /home3/p318770/venvs/simclr_vrt_gpu/venv311/bin/activate
cd /scratch/p318770/ssl_supcon_code && python norm_stats_zoo.py
