#!/bin/bash
#SBATCH --job-name=trtr_trained
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=trtr_trained_%j.out
# Train-vs-train null on the TRAINED ImageNet checkpoints (both schemes, eval-mode and train-mode BatchNorm):
# features of the standard 50,000 training images against 10,000 further training images (imagenet_trtr_extract.py),
# then the GPU tests with 10 seeds: MMD and ED (normalized) and Schilling/Henze at 1,500 and 10,000 per split.
# VRT, C2ST and MMDAgg follow in job_imgnet_trtr_trained_cpu.sh (submit it with --dependency=afterok on this job).
# Output root s50/trtr_trained, same layout as s50.
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source /home3/p318770/venvs/simclr_vrt_gpu/venv311/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
FEATS=/scratch/p318770/model_zoos/ladder_feats
BF=/home3/p318770/venvs/overfitting/benchmark_final/s50/trtr_trained
mkdir -p $BF/mmd_p2000 $BF/ed_p2000 $BF/knn_v2 $BF/vrt_fwd $BF/reversed $BF/c2st $BF/mmdagg
cd /scratch/p318770/ssl_supcon_code_s50
echo 'Extracting ImageNet to local SSD...'
time tar xf /scratch/public/ImageNet/imagenet.tar -C $TMPDIR
sync
echo 'Done.'
for MODE in eval train; do
  python -W ignore -u imagenet_trtr_extract.py --scheme both --bn_mode $MODE --workers 8
done
for TAG in imgnet_simple_trtr imgnet_adv_trtr imgnet_simple_bntrain_trtr imgnet_adv_bntrain_trtr; do
  NPZ="$FEATS/$TAG/ep{ep:03d}.npz"
  for N in 1500 10000; do
    python energy_tst_fast.py --tag $TAG --npz_fmt "$NPZ" --n_samples $N --n_perm 2000 --seeds 42-51 --epochs 1-300 --backend torch --out $BF/ed_p2000/ed_${TAG}_n${N}.csv
    python mmd_tst_fast.py    --tag $TAG --npz_fmt "$NPZ" --n_samples $N --n_perm 2000 --seeds 42-51 --epochs 1-300 --backend torch --out $BF/mmd_p2000/mmd_${TAG}_n${N}.csv
    python knn_tst_budget.py  --tag $TAG --npz_fmt "$NPZ" --n_samples $N --k 3 --n_perm 2000 --seeds 42-51 --epochs 1-300 --backend torch --out $BF/knn_v2/knn_${TAG}_n${N}_part.csv
  done
done
echo ALL DONE
