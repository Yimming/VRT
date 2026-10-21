#!/bin/bash
#SBATCH --job-name=base2
#SBATCH --partition=regular
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/p318770/ssl_supcon_code/base2_%a_%j.out
#SBATCH --array=0-5

module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source /home3/p318770/venvs/simclr_vrt_gpu/venv311/bin/activate

FEATS=/scratch/p318770/model_zoos/ladder_feats
OUT=/scratch/p318770/model_zoos/ladder_out

cd /scratch/p318770/ssl_supcon_code
echo "Start: $(date)  task=$SLURM_ARRAY_TASK_ID"

case $SLURM_ARRAY_TASK_ID in
  0) python baselines2_stats.py --model v7   --feats_dir $FEATS/v7 \
       --out_dir $OUT/base2_v7   --big_every 5 --c2st_paper ;;
  1) python baselines2_stats.py --model ablA --feats_dir $FEATS/ablA \
       --out_dir $OUT/base2_ablA --big_every 2 --c2st_paper ;;
  2) python baselines2_stats.py --model r50 --feats_dir $FEATS/r50 \
       --out_dir $OUT/base2_r50_c0 --epochs 0-50    --big_every 5 --c2st_paper ;;
  3) python baselines2_stats.py --model r50 --feats_dir $FEATS/r50 \
       --out_dir $OUT/base2_r50_c1 --epochs 51-100  --big_every 5 --c2st_paper ;;
  4) python baselines2_stats.py --model r50 --feats_dir $FEATS/r50 \
       --out_dir $OUT/base2_r50_c2 --epochs 101-150 --big_every 5 --c2st_paper ;;
  5) python baselines2_stats.py --model r50 --feats_dir $FEATS/r50 \
       --out_dir $OUT/base2_r50_c3 --epochs 151-200 --big_every 5 --c2st_paper ;;
esac
echo "End: $(date)"
