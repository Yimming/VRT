#!/bin/bash
#SBATCH --job-name=decomp
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --array=0-11
#SBATCH --output=decomp_%a_%j.out

module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source /home3/p318770/venvs/simclr_vrt_gpu/venv311/bin/activate

FEATS=/scratch/p318770/model_zoos/ladder_feats
OUT=/scratch/p318770/model_zoos/ladder_out

case $SLURM_ARRAY_TASK_ID in
  0)  python decomp_stats.py --model r50 --feats_dir $FEATS/r50 \
        --out_dir $OUT/decomp_r50 --epochs 40-140 --every 2 ;;
  1)  python decomp_stats.py --model v7 --feats_dir $FEATS/v7 \
        --out_dir $OUT/decomp_v7 --every 2 ;;
  2)  python decomp_stats.py --model ablA --feats_dir $FEATS/ablA \
        --out_dir $OUT/decomp_ablA ;;
  3)  python decomp_stats.py --model ablB --feats_dir $FEATS/ablB \
        --out_dir $OUT/decomp_ablB ;;
  4)  python decomp_stats.py --model ablC --feats_dir $FEATS/ablC \
        --out_dir $OUT/decomp_ablC ;;
  5)  python decomp_stats.py --model ablD --feats_dir $FEATS/ablD \
        --out_dir $OUT/decomp_ablD ;;
  6)  python decomp_stats.py --model wp --feats_dir $FEATS/wp \
        --out_dir $OUT/decomp_wp --every 20 ;;
  7)  python decomp_stats.py --model zoo_c10_m0 --feats_dir $FEATS/zoo_c10_m0 \
        --out_dir $OUT/decomp_zoo_c10_m0 ;;
  8)  python decomp_stats.py --model zoo_c10_m1 --feats_dir $FEATS/zoo_c10_m1 \
        --out_dir $OUT/decomp_zoo_c10_m1 ;;
  9)  python decomp_stats.py --model zoo_c100_m0 --feats_dir $FEATS/zoo_c100_m0 \
        --out_dir $OUT/decomp_zoo_c100_m0 ;;
  10) python decomp_stats.py --model zoo_c100_m1 --feats_dir $FEATS/zoo_c100_m1 \
        --out_dir $OUT/decomp_zoo_c100_m1 ;;
  11) python decomp_stats.py --model wp --feats_dir $FEATS/wp \
        --out_dir $OUT/decomp_wp2 --epochs 500-1000 --every 20 ;;
esac
