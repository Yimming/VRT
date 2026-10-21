#!/bin/bash
#SBATCH --job-name=profile
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-1
#SBATCH --output=profile_%a_%j.out

module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source /home3/p318770/venvs/simclr_vrt_gpu/venv311/bin/activate

FEATS=/scratch/p318770/model_zoos/ladder_feats
OUT=/scratch/p318770/model_zoos/ladder_out

case $SLURM_ARRAY_TASK_ID in
  0) python profile_diag.py --model zoo_c10_m0  --feats_dir $FEATS/zoo_c10_m0 \
       --out_dir $OUT/profile_zoo_c10_m0  --epochs 20,25,30,40,59 ;;
  1) python profile_diag.py --model zoo_c100_m0 --feats_dir $FEATS/zoo_c100_m0 \
       --out_dir $OUT/profile_zoo_c100_m0 --epochs 25,35,45,59 ;;
esac
