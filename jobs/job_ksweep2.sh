#!/bin/bash
#SBATCH --job-name=ksweep2
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-6
#SBATCH --output=ksweep2_%a_%j.out

module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source /home3/p318770/venvs/simclr_vrt_gpu/venv311/bin/activate

FEATS=/scratch/p318770/model_zoos/ladder_feats
OUT=/scratch/p318770/model_zoos/ladder_out

case $SLURM_ARRAY_TASK_ID in
  0) python ksweep_low.py --model v7   --feats_dir $FEATS/v7   --out_dir $OUT/ksweep_v7 --every 2 ;;
  1) python ksweep_low.py --model ablA --feats_dir $FEATS/ablA --out_dir $OUT/ksweep_ablA ;;
  2) python ksweep_low.py --model ablB --feats_dir $FEATS/ablB --out_dir $OUT/ksweep_ablB ;;
  3) python ksweep_low.py --model ablC --feats_dir $FEATS/ablC --out_dir $OUT/ksweep_ablC ;;
  4) python ksweep_low.py --model ablD --feats_dir $FEATS/ablD --out_dir $OUT/ksweep_ablD ;;
  5) python ksweep_low.py --model wp   --feats_dir $FEATS/wp   --out_dir $OUT/ksweep_wp --every 10 ;;
  6) python ksweep_low.py --model zoo_c10_m0 --feats_dir $FEATS/zoo_c10_m0 --out_dir $OUT/ksweep_zoo_c10_m0 ;;
esac
