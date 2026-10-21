#!/bin/bash
#SBATCH --job-name=adapt
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --array=0-3
#SBATCH --output=adapt_%a_%j.out

module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source /home3/p318770/venvs/simclr_vrt_gpu/venv311/bin/activate

FEATS=/scratch/p318770/model_zoos/ladder_feats
OUT=/scratch/p318770/model_zoos/ladder_out

case $SLURM_ARRAY_TASK_ID in
  0) python adaptive_theta.py --model r50 --feats_dir $FEATS/r50 \
       --out_dir $OUT/adapt_r50 --epochs 50-75 ;;
  1) python adaptive_theta.py --model zoo_c10_m0 --feats_dir $FEATS/zoo_c10_m0 \
       --out_dir $OUT/adapt_zoo_c10_m0 --epochs 15-40 ;;
  2) python adaptive_theta.py --model zoo_c100_m0 --feats_dir $FEATS/zoo_c100_m0 \
       --out_dir $OUT/adapt_zoo_c100_m0 --epochs 15-40 ;;
  3) python adaptive_theta.py --model ablC --feats_dir $FEATS/ablC \
       --out_dir $OUT/adapt_ablC --epochs 25-45 ;;
esac
