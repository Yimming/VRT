#!/bin/bash
#SBATCH --job-name=dec_fill
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=regular
#SBATCH --array=0-1
#SBATCH --output=/scratch/p318770/ssl_supcon_code/dec_fill_%a_%j.out
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source /home3/p318770/venvs/simclr_vrt_gpu/venv311/bin/activate
cd /scratch/p318770/ssl_supcon_code
FEATS=/scratch/p318770/model_zoos/ladder_feats
OUT=/scratch/p318770/model_zoos/ladder_out
case $SLURM_ARRAY_TASK_ID in
  0) python decomp_stats_odd.py --model r50 --feats_dir $FEATS/r50 \
       --out_dir $OUT/decomp_r50 --epochs 41-139 --every 2 ;;
  1) python decomp_stats_odd.py --model v7 --feats_dir $FEATS/v7 \
       --out_dir $OUT/decomp_v7 --every 2 ;;
esac
