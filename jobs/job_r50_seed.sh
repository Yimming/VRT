#!/bin/bash
#SBATCH --job-name=r50_seed
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/p318770/ssl_supcon_code/r50_seed%a_%j.out
#SBATCH --array=1-2

module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source /home3/p318770/venvs/simclr_vrt_gpu/venv311/bin/activate

cd /scratch/p318770/ssl_supcon_code
echo "Start: $(date)  seed=${SLURM_ARRAY_TASK_ID}"
python train_resnet50_cifar100_seed.py --seed ${SLURM_ARRAY_TASK_ID}
echo "End: $(date)"
