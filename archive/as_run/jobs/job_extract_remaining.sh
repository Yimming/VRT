#!/bin/bash
# ONE job: extract raw features for ALL remaining Table-1 models
# (ablB/C/D + WildPark + Phase Zoo x3 + Zoo C10 x5 + Zoo C100 x5).
# r50/v7/ablA are handled by the already-queued job 30259556.
#SBATCH --job-name=ext_rem
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/p318770/ssl_supcon_code/ext_rem_%j.out

module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source /home3/p318770/venvs/simclr_vrt_gpu/venv311/bin/activate

cd /scratch/p318770/ssl_supcon_code
echo "Start: $(date)"
for M in ablB ablC ablD wp phase zoo_c10 zoo_c100; do
  echo "== extracting $M =="
  python extract_feats.py --model $M
done
echo "End: $(date)"
