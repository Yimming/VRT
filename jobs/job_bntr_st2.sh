#!/bin/bash
#SBATCH --job-name=bnst2
#SBATCH --partition=regular
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --array=0-11
#SBATCH --output=/scratch/p318770/ssl_supcon_code/bnst2_%A_%a.out
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source /home3/p318770/venvs/simclr_vrt_gpu/venv311/bin/activate
cd /scratch/p318770/ssl_supcon_code
MODELS=(ablA ablC zoo_c10_m0 zoo_c10_m1 zoo_c10_m2 zoo_c10_m3 zoo_c10_m4 \
        zoo_c100_m0 zoo_c100_m1 zoo_c100_m2 zoo_c100_m3 zoo_c100_m4)
M=${MODELS[$SLURM_ARRAY_TASK_ID]}
F=/scratch/p318770/model_zoos/ladder_feats
O=/scratch/p318770/model_zoos/ladder_out
python ladder_stats.py --model ${M}_bntrain --feats_dir $F/${M}_bntrain --out_dir $O
