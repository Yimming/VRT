#!/bin/bash
#SBATCH --job-name=imgnet_extbn
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=imgnet_extbn_%j.out
# Section 4.4 check on the camera-ready ImageNet features: re-extract the same 50,000 + 10,000 images
# (same seeds 42/123, verified against the eval-mode files) with train-mode BatchNorm, i.e. every BN layer
# normalizes with the statistics of the current 256-image batch instead of the running statistics.
# Output: /scratch/p318770/model_zoos/ladder_feats/imgnet_{simple,adv}_bntrain/ep{ep:03d}.npz
# The first attempt (48G, v100) was OOM-killed right after the tar step: Slurm's cgroup accounting counts the
# page cache, which the 158 GB extraction fills to the limit. sync flushes the dirty pages before Python starts.
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source /home3/p318770/venvs/simclr_vrt_gpu/venv311/bin/activate
echo 'Extracting ImageNet to local SSD...'
time tar xf /scratch/public/ImageNet/imagenet.tar -C $TMPDIR
sync
echo 'Done.'
cd /scratch/p318770/ssl_supcon_code
python -W ignore -u imagenet_extract.py --scheme both --workers 6 --bn_mode train
