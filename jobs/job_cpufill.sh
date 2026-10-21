#!/bin/bash
#SBATCH --job-name=cpufill
#SBATCH --partition=regular
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --array=0-39
#SBATCH --output=/scratch/p318770/ssl_supcon_code/cpufill_%a_%j.out
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source /home3/p318770/venvs/simclr_vrt_gpu/venv311/bin/activate
cd /scratch/p318770/ssl_supcon_code
F=/scratch/p318770/model_zoos/ladder_feats
O=/scratch/p318770/model_zoos/ladder_out
METHODS=(c2st mmdagg)
UNITS=(r50_c0 r50_c1 r50_c2 r50_c3 v7a v7b ablA ablB ablC ablD \
       zoo_c10_m0 zoo_c10_m1 zoo_c10_m2 zoo_c10_m3 zoo_c10_m4 \
       zoo_c100_m0 zoo_c100_m1 zoo_c100_m2 zoo_c100_m3 zoo_c100_m4)
FEATS=(r50 r50 r50 r50 v7 v7 ablA ablB ablC ablD \
       zoo_c10_m0 zoo_c10_m1 zoo_c10_m2 zoo_c10_m3 zoo_c10_m4 \
       zoo_c100_m0 zoo_c100_m1 zoo_c100_m2 zoo_c100_m3 zoo_c100_m4)
EPOCHS=(0-50 51-100 101-150 151-200 0-99 100-199 0-49 0-49 0-49 0-49 \
        0-50 0-50 0-50 0-50 0-50 0-60 0-60 0-60 0-60 0-60)
M=${METHODS[$((SLURM_ARRAY_TASK_ID / 20))]}
i=$((SLURM_ARRAY_TASK_ID % 20))
python bigfill.py --method $M --unit ${UNITS[$i]} --feats_dir $F/${FEATS[$i]} \
    --epochs ${EPOCHS[$i]} --out_dir $O/bigfill_${UNITS[$i]}
