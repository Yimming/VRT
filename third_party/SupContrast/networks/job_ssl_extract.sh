#!/bin/bash
#SBATCH --job-name=ssl_ext
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --output=/scratch/p318770/ssl_supcon_code/ssl_ext_%a_%j.out
#SBATCH --array=0-1

module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source /home3/p318770/venvs/simclr_vrt_gpu/venv311/bin/activate

SAVE=/scratch/p318770/model_zoos/ssl_supcon
cd /scratch/p318770/ssl_supcon_code
echo "Start: $(date)  task=$SLURM_ARRAY_TASK_ID"

# VRT-first pass: no --with_mmd (MMD is backfilled later from the saved
# fp16 features on CPU; the naive in-loop MMD was the bottleneck)
case $SLURM_ARRAY_TASK_ID in
  0) python supcon_extract_vrt.py --dataset cifar10 --every 10 --save_feats \
       --ckpt_dir $SAVE/SimCLR_cifar10_resnet50_lr_0.25_decay_0.0001_bsz_512_temp_0.5_trial_ssl_c10_s0_cosine_warm ;;
  1) python supcon_extract_vrt.py --dataset cifar100 --every 10 --save_feats \
       --ckpt_dir $SAVE/SimCLR_cifar100_resnet50_lr_0.25_decay_0.0001_bsz_512_temp_0.5_trial_ssl_c100_s0_cosine_warm ;;
esac
echo "End: $(date)"
