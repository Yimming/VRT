#!/bin/bash
#SBATCH --job-name=fillvrt
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=1-4
#SBATCH --output=fillvrt_%a_%j.out
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source /home3/p318770/venvs/simclr_vrt_gpu/venv311/bin/activate
M=$SLURM_ARRAY_TASK_ID
python ksweep_low.py --model zoo_c100_m$M \
  --feats_dir /scratch/p318770/model_zoos/ladder_feats/zoo_c100_m$M \
  --out_dir /scratch/p318770/model_zoos/ladder_out/ksweep_zoo_c100_m$M
