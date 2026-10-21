#!/bin/bash
#SBATCH --job-name=imgnet_rand
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=imgnet_rand_%j.out
# Random-initialization ImageNet null (Appendix C protocol) for every method: 30 random DeepResNet50 inits,
# eval-mode and train-mode BatchNorm, same 50,000 + 10,000 images as all other ImageNet runs, one analysis seed
# per init. Then the fast tests on both folders (ED, MMD normalized and raw, Schilling/Henze, forward and reversed
# VRT at both budgets, C2ST at both budgets). MMDAgg is the separate CPU job job_imgnet_randinit_mmdagg.sh.
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source /home3/p318770/venvs/simclr_vrt_gpu/venv311/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
FEATS=/scratch/p318770/model_zoos/ladder_feats
BF=/home3/p318770/venvs/overfitting/benchmark_final/s50/null
mkdir -p $BF/mmd_p2000 $BF/mmd_raw_p2000 $BF/ed_p2000 $BF/ed_raw_p2000 $BF/knn_v2 $BF/vrt_fwd $BF/reversed $BF/c2st $BF/mmdagg
cd /scratch/p318770/ssl_supcon_code_s50
echo 'Extracting ImageNet to local SSD...'
time tar xf /scratch/public/ImageNet/imagenet.tar -C $TMPDIR
sync
echo 'Done.'
for MODE in eval train; do
  python -W ignore -u imagenet_randinit_extract.py --n_init 30 --bn_mode $MODE --workers 8
done
for MODE in eval train; do
  TAG=imgnet_rand_$MODE; NPZ="$FEATS/$TAG/ep{ep:03d}.npz"
  for N in 1500 10000; do
    python energy_tst_fast.py --tag $TAG --npz_fmt "$NPZ" --n_samples $N --n_perm 2000 --seeds 42 --epochs 0-29 --backend torch       --out $BF/ed_p2000/ed_${TAG}_n${N}.csv
    python energy_tst_fast.py --tag $TAG --npz_fmt "$NPZ" --n_samples $N --n_perm 2000 --seeds 42 --epochs 0-29 --backend torch --raw --out $BF/ed_raw_p2000/ed_${TAG}_n${N}.csv
    python mmd_tst_fast.py    --tag $TAG --npz_fmt "$NPZ" --n_samples $N --n_perm 2000 --seeds 42 --epochs 0-29 --backend torch       --out $BF/mmd_p2000/mmd_${TAG}_n${N}.csv
    python mmd_tst_fast.py    --tag $TAG --npz_fmt "$NPZ" --n_samples $N --n_perm 2000 --seeds 42 --epochs 0-29 --backend torch --raw --out $BF/mmd_raw_p2000/mmd_${TAG}_n${N}.csv
    python knn_tst_budget.py  --tag $TAG --npz_fmt "$NPZ" --n_samples $N --k 3 --n_perm 2000 --seeds 42 --epochs 0-29 --backend torch --out $BF/knn_v2/knn_${TAG}_n${N}_part.csv
  done
  for B in large small; do
    python vrt_forward_budget.py  --tag $TAG --budget $B --npz_fmt "$NPZ" --epochs 0-29 --seeds 42 --n_perm 2000 --out $BF/vrt_fwd/vrt_fwd_${TAG}_${B}.csv
    python vrt_reversed_budget.py --tag $TAG --budget $B --npz_fmt "$NPZ" --epochs 0-29 --seeds 42 --n_perm 2000 --out $BF/reversed/vrt_rev_${TAG}_${B}.csv
  done
  python base_fill.py --stat c2st --tag $TAG --npz_fmt "$NPZ" --budgets 1500,10000 --every 1 --seeds 42 --epochs 0-29 --out_dir $BF/c2st
done
echo ALL DONE
