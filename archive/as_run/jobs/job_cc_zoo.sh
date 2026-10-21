#!/bin/bash
#SBATCH --job-name=cc_zoo
#SBATCH --partition=regular
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/p318770/ssl_supcon_code/cc_zoo_%j.out
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source /home3/p318770/venvs/simclr_vrt_gpu/venv311/bin/activate
cd /scratch/p318770/ssl_supcon_code
F=/scratch/p318770/model_zoos/ladder_feats
O=/scratch/p318770/model_zoos/ladder_out
for i in 0 1 2 3 4; do
  python cc_tests.py --model zoo_c10_m${i}_traj  --dataset cifar10 \
      --feats_dir $F/zoo_c10_m${i}  --out_dir $O --n_perm 999
  python cc_tests.py --model zoo_c100_m${i}_traj --dataset cifar100 \
      --feats_dir $F/zoo_c100_m${i} --out_dir $O --n_perm 999
done
