#!/bin/bash
#SBATCH --job-name=sslvrtlow
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-15
#SBATCH --output=/scratch/p318770/ssl_supcon_code/sslvrtlow_%a_%j.out
# Small-budget VRT (k=15 protocol of the rebuttal) on the SimCLR models, every epoch 1-1000, features on the fly.
# 16 tasks x 125 epochs: tasks 0-7 CIFAR-10, tasks 8-15 CIFAR-100.
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source /home3/p318770/venvs/simclr_vrt_gpu/venv311/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd /scratch/p318770/ssl_supcon_code
i=$SLURM_ARRAY_TASK_ID
if [ $i -lt 8 ]; then
  DS=cifar10;  CK=$(ls -d /scratch/p318770/model_zoos/ssl_supcon/SimCLR_cifar10_* | head -1);  C=$i
else
  DS=cifar100; CK=$(ls -d /scratch/p318770/model_zoos/ssl_supcon/SimCLR_cifar100_* | head -1); C=$((i - 8))
fi
LO=$((C * 125 + 1)); HI=$(((C + 1) * 125))
python ssl_vrt_low_fill.py --ckpt_dir $CK --dataset $DS --lo $LO --hi $HI
