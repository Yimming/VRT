#!/bin/bash
#SBATCH --job-name=sslfill
#SBATCH --partition=gpushort
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=03:59:00
#SBATCH --array=0-19
#SBATCH --output=/scratch/p318770/ssl_supcon_code/sslfill_%a_%j.out
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source /home3/p318770/venvs/simclr_vrt_gpu/venv311/bin/activate
cd /scratch/p318770/ssl_supcon_code
i=$SLURM_ARRAY_TASK_ID
if [ $i -lt 10 ]; then
  DS=cifar10
  CK=$(ls -d /scratch/p318770/model_zoos/ssl_supcon/SimCLR_cifar10_* | head -1)
  C=$i
else
  DS=cifar100
  CK=$(ls -d /scratch/p318770/model_zoos/ssl_supcon/SimCLR_cifar100_* | head -1)
  C=$((i - 10))
fi
LO=$((C * 100 + 1))
HI=$(((C + 1) * 100))
python supcon_fill_vrt.py --ckpt_dir $CK --dataset $DS --lo $LO --hi $HI
