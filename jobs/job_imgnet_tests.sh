#!/bin/bash
#SBATCH --job-name=imgnet_tests
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=imgnet_tests_%j.out
# All baselines on the saved ImageNet features (two schemes x 9 checkpoints), unified protocol:
# MMD / ED at 1,500 and 10,000 (2,000 permutations), C2ST 1,500 / 10,000, MMDAgg 1,500 / 5,000,
# Schilling / Henze at 1,500 and 10,000, reversed VRT at both budgets; 10 seeds everywhere.
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source /home3/p318770/venvs/simclr_vrt_gpu/venv311/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
FEATS=/scratch/p318770/model_zoos/ladder_feats
BF=/home3/p318770/venvs/overfitting/benchmark_final
cd /scratch/p318770/ssl_supcon_code
EPOCHS=1,5,20,50,100,150,200,250,300
for TAG in imgnet_simple imgnet_adv; do
  NPZ="$FEATS/$TAG/ep{ep:03d}.npz"
  for N in 1500 10000; do
    python energy_tst_fast.py --tag $TAG --npz_fmt "$NPZ" --n_samples $N --n_perm 2000 --seeds 42-51 --epochs $EPOCHS --backend torch --out $BF/ed_p2000/ed_${TAG}_n${N}.csv
    python mmd_tst_fast.py    --tag $TAG --npz_fmt "$NPZ" --n_samples $N --n_perm 2000 --seeds 42-51 --epochs $EPOCHS --backend torch --out $BF/mmd_p2000/mmd_${TAG}_n${N}.csv
    python knn_tst_budget.py  --tag $TAG --npz_fmt "$NPZ" --n_samples $N --k 3 --n_perm 2000 --seeds 42-51 --epochs $EPOCHS --backend torch --out $BF/knn_v2/knn_${TAG}_n${N}_part${TAG}.csv
  done
  python base_fill.py --stat c2st   --tag $TAG --npz_fmt "$NPZ" --budgets 1500,10000 --every 1 --seeds 42-51 --epochs $EPOCHS --out_dir $BF/c2st_10seed
  python base_fill.py --stat mmdagg --tag $TAG --npz_fmt "$NPZ" --budgets 1500      --every 1 --seeds 42-51 --epochs $EPOCHS --out_dir $BF/mmdagg_10seed
  python base_fill.py --stat mmdagg --tag $TAG --npz_fmt "$NPZ" --budgets 5000      --every 1 --seeds 42-51 --epochs $EPOCHS --out_dir $BF/mmdagg_10seed
  for B in large small; do
    python vrt_reversed_budget.py --tag $TAG --budget $B --npz_fmt "$NPZ" --epochs $EPOCHS --seeds 42-51 --n_perm 2000 --out $BF/reversed/vrt_rev_${TAG}_${B}.csv
  done
done
echo ALL DONE
