#!/bin/bash
#SBATCH --job-name=ksweep
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-1
#SBATCH --output=ksweep_%a_%j.out

module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source /home3/p318770/venvs/simclr_vrt_gpu/venv311/bin/activate

FEATS=/scratch/p318770/model_zoos/ladder_feats
OUT=/scratch/p318770/model_zoos/ladder_out

case $SLURM_ARRAY_TASK_ID in
  0) python ksweep_low.py --model r50 --feats_dir $FEATS/r50 \
       --out_dir $OUT/ksweep_r50 --epochs 50-130 --every 5 ;;
  1) python ksweep_low.py --model zoo_c100_m0 --feats_dir $FEATS/zoo_c100_m0 \
       --out_dir $OUT/ksweep_zoo_c100_m0 --every 2 ;;
esac
