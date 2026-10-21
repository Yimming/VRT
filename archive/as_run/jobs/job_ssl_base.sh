#!/bin/bash
#SBATCH --job-name=sslbase
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-39
#SBATCH --output=/scratch/p318770/ssl_supcon_code/sslbase_%a_%j.out
# Per-epoch baselines on the two SimCLR models (MMD, ED, C2ST, MMDAgg; 10 seeds; 2,000 permutations),
# features extracted on the fly from the checkpoints as in job_sslfill.sh. 40 tasks x 50 epochs:
# tasks 0-19 CIFAR-10 epochs 1-1000, tasks 20-39 CIFAR-100 epochs 1-1000.
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source /home3/p318770/venvs/simclr_vrt_gpu/venv311/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd /scratch/p318770/ssl_supcon_code
i=$SLURM_ARRAY_TASK_ID
if [ $i -lt 20 ]; then
  DS=cifar10
  CK=$(ls -d /scratch/p318770/model_zoos/ssl_supcon/SimCLR_cifar10_* | head -1)
  C=$i
else
  DS=cifar100
  CK=$(ls -d /scratch/p318770/model_zoos/ssl_supcon/SimCLR_cifar100_* | head -1)
  C=$((i - 20))
fi
LO=$((C * 50 + 1))
HI=$(((C + 1) * 50))
python ssl_baselines_fill.py --ckpt_dir $CK --dataset $DS --lo $LO --hi $HI
