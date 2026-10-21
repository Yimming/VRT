#!/bin/bash
#SBATCH --job-name=bntr_st
#SBATCH --partition=regular
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=16:00:00
#SBATCH --output=/scratch/p318770/ssl_supcon_code/bntr_st_%j.out
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source /home3/p318770/venvs/simclr_vrt_gpu/venv311/bin/activate
cd /scratch/p318770/ssl_supcon_code
F=/scratch/p318770/model_zoos/ladder_feats
O=/scratch/p318770/model_zoos/ladder_out
python ladder_stats.py --model r50_bntrain  --feats_dir $F/r50_bntrain  --out_dir $O --epochs 0-200 --mmd_full_every 5
python ladder_stats.py --model ablB_bntrain --feats_dir $F/ablB_bntrain --out_dir $O
python ladder_stats.py --model ablD_bntrain --feats_dir $F/ablD_bntrain --out_dir $O
