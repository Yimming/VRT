#!/bin/bash
#SBATCH --job-name=sslagg5kr
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=/scratch/p318770/ssl_supcon_code/sslagg5kr_%j.out
# Refinement run for MMDAgg 5,000 on SimCLR: sbatch job_ssl_mmdagg5k_range.sh <cifar10|cifar100> <lo> <hi> <every>
DS=$1; LO=$2; HI=$3; EV=$4
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source /home3/p318770/venvs/simclr_vrt_gpu/venv311/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd /scratch/p318770/ssl_supcon_code
if [ "$DS" = "cifar10" ]; then CK=$(ls -d /scratch/p318770/model_zoos/ssl_supcon/SimCLR_cifar10_* | head -1); else CK=$(ls -d /scratch/p318770/model_zoos/ssl_supcon/SimCLR_cifar100_* | head -1); fi
python ssl_mmdagg5k_fill.py --ckpt_dir $CK --dataset $DS --lo $LO --hi $HI --every $EV
