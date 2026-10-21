#!/bin/bash
#SBATCH --job-name=cc_r50end
#SBATCH --partition=regular
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/p318770/ssl_supcon_code/cc_r50end_%j.out
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source /home3/p318770/venvs/simclr_vrt_gpu/venv311/bin/activate
cd /scratch/p318770/ssl_supcon_code
F=/scratch/p318770/model_zoos/ladder_feats
O=/scratch/p318770/model_zoos/ladder_out
LAST=$(ls $F/r50/ep*.npz | sed 's/.*ep0*\([0-9]*\)\.npz/\1/' | sort -n | tail -1)
echo "last r50 feature epoch: $LAST"
python cc_tests.py --model r50 --dataset cifar100 --feats_dir $F/r50 \
    --out_dir $O --epochs $LAST --n_perm 9999
mkdir -p $O/cc_tail_$LAST
python cc_tests.py --model r50_traj --dataset cifar100 --feats_dir $F/r50 \
    --out_dir $O/cc_tail_$LAST --epochs $LAST --n_perm 999
