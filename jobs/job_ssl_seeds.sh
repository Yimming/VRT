#!/bin/bash
#SBATCH --job-name=sslseeds
#SBATCH --array=0-7
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/p318770/ssl_supcon_code/sslseeds_%a_%j.out
# Seed-robustness check for the SimCLR large-budget onsets: 40 extra analysis seeds (52-91) for VRT, MMD and
# ED at 10,000 per split on epochs 170-300 of both SimCLR models, same code paths as the published numbers.
# Tasks 0-3 CIFAR-10, 4-7 CIFAR-100, about 33 epochs each (roughly 8 minutes per epoch on a GPU node).
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
RANGES=("170 202" "203 235" "236 268" "269 300")
set -- ${RANGES[$C]}
echo "task $i: $DS epochs $1-$2 ($CK)"
python ssl_seeds_fill.py --ckpt_dir $CK --dataset $DS --lo $1 --hi $2 --seeds 52-91 --backend torch
echo DONE
