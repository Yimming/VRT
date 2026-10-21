#!/bin/bash
#SBATCH --job-name=null10_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=null10_gpu_%j.out
# Ten analysis seeds per random init (42-51) instead of one, on the saved random-initialization features:
# train-vs-validation (imgnet_rand_eval / _train) and train-vs-train (_trtr). GPU part: MMD and ED (normalized),
# Schilling/Henze, at 1,500 and 10,000 per split. Output root s50/null10, same layout as s50/null.
# With one fixed analysis seed the 30 inits share one reference and subsample draw and are not independent trials;
# ten seeds per init give the rate averaged over draws and the paper's median-over-seeds decision per init.
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source /home3/p318770/venvs/simclr_vrt_gpu/venv311/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
FEATS=/scratch/p318770/model_zoos/ladder_feats
BF=/home3/p318770/venvs/overfitting/benchmark_final/s50/null10
mkdir -p $BF/mmd_p2000 $BF/ed_p2000 $BF/knn_v2 $BF/vrt_fwd $BF/reversed $BF/c2st $BF/mmdagg
cd /scratch/p318770/ssl_supcon_code_s50
for TAG in imgnet_rand_eval imgnet_rand_train imgnet_rand_eval_trtr imgnet_rand_train_trtr; do
  NPZ="$FEATS/$TAG/ep{ep:03d}.npz"
  for N in 1500 10000; do
    python energy_tst_fast.py --tag $TAG --npz_fmt "$NPZ" --n_samples $N --n_perm 2000 --seeds 42-51 --epochs 0-29 --backend torch --out $BF/ed_p2000/ed_${TAG}_n${N}.csv
    python mmd_tst_fast.py    --tag $TAG --npz_fmt "$NPZ" --n_samples $N --n_perm 2000 --seeds 42-51 --epochs 0-29 --backend torch --out $BF/mmd_p2000/mmd_${TAG}_n${N}.csv
    python knn_tst_budget.py  --tag $TAG --npz_fmt "$NPZ" --n_samples $N --k 3 --n_perm 2000 --seeds 42-51 --epochs 0-29 --backend torch --out $BF/knn_v2/knn_${TAG}_n${N}_part.csv
  done
done
# the one-seed MMDAgg results stay as they are (300 s per test); copied here so that one harvest shows everything
cp -n /home3/p318770/venvs/overfitting/benchmark_final/s50/null/mmdagg/*.csv $BF/mmdagg/ 2>/dev/null
cp -n /home3/p318770/venvs/overfitting/benchmark_final/s50/null_trtr/mmdagg/*.csv $BF/mmdagg/ 2>/dev/null
echo ALL DONE
