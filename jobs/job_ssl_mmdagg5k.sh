#!/bin/bash
#SBATCH --job-name=sslagg5k
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=0-7
#SBATCH --output=/scratch/p318770/ssl_supcon_code/sslagg5k_%a_%j.out
# MMDAgg at 5,000 per split on the SimCLR models, ROUND 1: coarse grid, every 50th epoch, CPU nodes.
# 8 tasks x 250 epochs (5 evaluation points each): tasks 0-3 CIFAR-10, tasks 4-7 CIFAR-100.
# ROUND 2 (after harvest_ssl.py shows the coarse onset interval): per-epoch inside the 50-epoch interval and
# every 25th epoch after it, with  sbatch job_ssl_mmdagg5k_range.sh <dataset> <lo> <hi> <every>
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source /home3/p318770/venvs/simclr_vrt_gpu/venv311/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd /scratch/p318770/ssl_supcon_code
i=$SLURM_ARRAY_TASK_ID
if [ $i -lt 4 ]; then
  DS=cifar10;  CK=$(ls -d /scratch/p318770/model_zoos/ssl_supcon/SimCLR_cifar10_* | head -1);  C=$i
else
  DS=cifar100; CK=$(ls -d /scratch/p318770/model_zoos/ssl_supcon/SimCLR_cifar100_* | head -1); C=$((i - 4))
fi
LO=$((C * 250 + 1)); HI=$(((C + 1) * 250))
python ssl_mmdagg5k_fill.py --ckpt_dir $CK --dataset $DS --lo $LO --hi $HI --every 50
