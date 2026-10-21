#!/bin/bash
#SBATCH --job-name=cc_close
#SBATCH --partition=regular
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-30
#SBATCH --output=/scratch/p318770/ssl_supcon_code/cc_close_%a_%j.out
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source /home3/p318770/venvs/simclr_vrt_gpu/venv311/bin/activate
cd /scratch/p318770/ssl_supcon_code
F=/scratch/p318770/model_zoos/ladder_feats
O=/scratch/p318770/model_zoos/ladder_out
i=$SLURM_ARRAY_TASK_ID
if [ $i -lt 20 ]; then
  EP=$((180 + i))
  mkdir -p $O/cc_tail_$EP
  python cc_tests.py --model r50_traj --dataset cifar100 --feats_dir $F/r50 \
      --out_dir $O/cc_tail_$EP --epochs $EP --n_perm 999
elif [ $i -eq 20 ]; then
  python cc_tests.py --model r50 --dataset cifar100 --feats_dir $F/r50 \
      --out_dir $O --epochs 199 --n_perm 9999
else
  j=$((i - 21))
  if [ $j -lt 5 ]; then
    M=zoo_c10_m$j; DS=cifar10; EP=50
  else
    M=zoo_c100_m$((j - 5)); DS=cifar100; EP=60
  fi
  python cc_tests.py --model $M --dataset $DS --feats_dir $F/$M \
      --out_dir $O --epochs $EP --n_perm 9999
fi
