#!/bin/bash
#SBATCH --job-name=sslfill
#SBATCH --partition=gpushort
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --array=0-11
#SBATCH --output=/scratch/p318770/ssl_supcon_code/sslfill2_%a_%j.out
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source /home3/p318770/venvs/simclr_vrt_gpu/venv311/bin/activate
cd /scratch/p318770/ssl_supcon_code
CK=$(ls -d /scratch/p318770/model_zoos/ssl_supcon/SimCLR_cifar100_* | head -1)
i=$SLURM_ARRAY_TASK_ID
LO=$((401 + i * 50))
HI=$((LO + 49))
python supcon_fill_vrt.py --ckpt_dir $CK --dataset cifar100 --lo $LO --hi $HI
