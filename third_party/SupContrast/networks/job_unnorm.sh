#!/bin/bash
#SBATCH --job-name=unnorm
#SBATCH --partition=regular
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/p318770/ssl_supcon_code/unnorm_%a_%j.out
#SBATCH --array=0-5

module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source /home3/p318770/venvs/simclr_vrt_gpu/venv311/bin/activate

FEATS=/scratch/p318770/model_zoos/ladder_feats
OUT=/scratch/p318770/model_zoos/ladder_out

cd /scratch/p318770/ssl_supcon_code
echo "Start: $(date)  task=$SLURM_ARRAY_TASK_ID"

case $SLURM_ARRAY_TASK_ID in
  0) python unnorm_stats.py --model v7   --feats_dir $FEATS/v7  \
       --out_dir $OUT/unnorm_v7  --mmd10k_every 5 ;;
  1) python unnorm_stats.py --model ablA --feats_dir $FEATS/ablA \
       --out_dir $OUT/unnorm_ablA --mmd10k_every 2 ;;
  2) python unnorm_stats.py --model r50 --feats_dir $FEATS/r50 \
       --out_dir $OUT/unnorm_r50_c0 --epochs 0-50    --mmd10k_every 5 ;;
  3) python unnorm_stats.py --model r50 --feats_dir $FEATS/r50 \
       --out_dir $OUT/unnorm_r50_c1 --epochs 51-100  --mmd10k_every 5 ;;
  4) python unnorm_stats.py --model r50 --feats_dir $FEATS/r50 \
       --out_dir $OUT/unnorm_r50_c2 --epochs 101-150 --mmd10k_every 5 ;;
  5) python unnorm_stats.py --model r50 --feats_dir $FEATS/r50 \
       --out_dir $OUT/unnorm_r50_c3 --epochs 151-200 --mmd10k_every 5 ;;
esac
echo "End: $(date)"
