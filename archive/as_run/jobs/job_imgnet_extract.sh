#!/bin/bash
#SBATCH --job-name=imgnet_ext
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=imgnet_ext_%j.out
# Extract the full ImageNet (1,281,167 train / 50,000 val) to local SSD exactly as the paper-era job did,
# then save the DeepResNet50 features (same seeds 42/123, same 50,000 + 10,000 images) as npz for both schemes.
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source /home3/p318770/venvs/simclr_vrt_gpu/venv311/bin/activate
echo 'Extracting ImageNet to local SSD...'
time tar xf /scratch/public/ImageNet/imagenet.tar -C $TMPDIR
echo 'Done.'
cd /scratch/p318770/ssl_supcon_code
python -W ignore -u imagenet_extract.py --scheme both --workers 8
