#!/bin/bash
#SBATCH --job-name=sslrange
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/p318770/ssl_supcon_code/sslrange_%j.out
# Per-epoch SimCLR baselines for an arbitrary epoch range, to fill gaps in parallel.
# Usage: sbatch job_ssl_range.sh <cifar10|cifar100> <lo> <hi>
# Writes sslbase_<dataset>_<lo>-<hi>.csv (same layout, picked up by harvest_ssl.py automatically).
DS=$1; LO=$2; HI=$3
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source /home3/p318770/venvs/simclr_vrt_gpu/venv311/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd /scratch/p318770/ssl_supcon_code
if [ "$DS" = "cifar10" ]; then
  CK=$(ls -d /scratch/p318770/model_zoos/ssl_supcon/SimCLR_cifar10_* | head -1)
else
  CK=$(ls -d /scratch/p318770/model_zoos/ssl_supcon/SimCLR_cifar100_* | head -1)
fi
python ssl_baselines_fill.py --ckpt_dir $CK --dataset $DS --lo $LO --hi $HI
