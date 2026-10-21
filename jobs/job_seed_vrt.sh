#!/bin/bash
# Multi-seed VRT reproduction (reviewer Ftg8): extract features for the
# R50 seed reruns and run the ladder on the onset region (epochs 20-140).
# GPU part (extraction) and CPU part (stats) combined per seed for
# simplicity; one array task per seed.
#SBATCH --job-name=seed_vrt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --output=/scratch/p318770/ssl_supcon_code/seed_vrt_%a_%j.out
#SBATCH --array=1,2,4

module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source /home3/p318770/venvs/simclr_vrt_gpu/venv311/bin/activate

S=$SLURM_ARRAY_TASK_ID
FEATS=/scratch/p318770/model_zoos/ladder_feats
OUT=/scratch/p318770/model_zoos/ladder_out

cd /scratch/p318770/ssl_supcon_code
echo "Start: $(date)  seed=$S"
python extract_feats.py --model r50 \
  --r50_dir /scratch/p318770/model_zoos/resnet50_cifar100_seed$S \
  --out_name r50_seed$S
python ladder_stats.py --model r50_seed$S \
  --feats_dir $FEATS/r50_seed$S --out_dir $OUT/ladder_r50_seed$S \
  --epochs 20-140
echo "End: $(date)"
