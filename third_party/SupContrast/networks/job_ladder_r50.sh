#!/bin/bash
#SBATCH --job-name=ladder_r50
#SBATCH --partition=regular
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/p318770/ssl_supcon_code/ladder_r50_%a_%j.out
#SBATCH --array=0-3

module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source /home3/p318770/venvs/simclr_vrt_gpu/venv311/bin/activate

RANGES=("0-50" "51-100" "101-150" "151-200")
R=${RANGES[$SLURM_ARRAY_TASK_ID]}

cd /scratch/p318770/ssl_supcon_code
echo "Start: $(date)  range=$R"
python ladder_stats.py --model r50 \
  --feats_dir /scratch/p318770/model_zoos/ladder_feats/r50 \
  --out_dir /scratch/p318770/model_zoos/ladder_out/r50_chunk${SLURM_ARRAY_TASK_ID} \
  --epochs $R
echo "End: $(date)"
