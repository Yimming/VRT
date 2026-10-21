#!/bin/bash
#SBATCH --job-name=imgnet_bntests
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=imgnet_bntests_%j.out
# Baselines on the train-mode-BatchNorm ImageNet features (Section 4.4 check), unified protocol, 10 seeds,
# 2,000 permutations: MMD / ED / Schilling-Henze at 1,500 and 10,000, MMDAgg at 1,500, small-budget VRT
# (ksweep15_fill.py, unchanged rebuttal script) and large-budget forward VRT. MMDAgg 5,000 is the separate
# CPU array job_imgnet_bntrain_mmdagg5k.sh. The forward large-budget VRT is also run on the eval-mode
# features, so the ImageNet VRT cell is on the re-extracted features like every other cell.
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source /home3/p318770/venvs/simclr_vrt_gpu/venv311/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
FEATS=/scratch/p318770/model_zoos/ladder_feats
BF=/home3/p318770/venvs/overfitting/benchmark_final
mkdir -p $BF/vrt_fwd $BF/ksweep15_extra
cd /scratch/p318770/ssl_supcon_code
python -c "import mmdagg" || { echo "mmdagg not importable, abort"; exit 1; }
EPOCHS=1,5,20,50,100,150,200,250,300
for TAG in imgnet_simple_bntrain imgnet_adv_bntrain; do
  NPZ="$FEATS/$TAG/ep{ep:03d}.npz"
  ls $FEATS/$TAG/ep*.npz | wc -l
  for N in 1500 10000; do
    python energy_tst_fast.py --tag $TAG --npz_fmt "$NPZ" --n_samples $N --n_perm 2000 --seeds 42-51 --epochs $EPOCHS --backend torch --out $BF/ed_p2000/ed_${TAG}_n${N}.csv
    python mmd_tst_fast.py    --tag $TAG --npz_fmt "$NPZ" --n_samples $N --n_perm 2000 --seeds 42-51 --epochs $EPOCHS --backend torch --out $BF/mmd_p2000/mmd_${TAG}_n${N}.csv
    python knn_tst_budget.py  --tag $TAG --npz_fmt "$NPZ" --n_samples $N --k 3 --n_perm 2000 --seeds 42-51 --epochs $EPOCHS --backend torch --out $BF/knn_v2/knn_${TAG}_n${N}_part${TAG}.csv
  done
  python vrt_forward_budget.py --tag $TAG --budget large --npz_fmt "$NPZ" --epochs $EPOCHS --seeds 42-51 --n_perm 2000 --out $BF/vrt_fwd/vrt_fwd_${TAG}_large.csv
  python ksweep15_fill.py --unit $TAG --feats_dir $FEATS/$TAG --out_dir $BF/ksweep15_extra --lo 0 --hi 1000
  python base_fill.py --stat mmdagg --tag $TAG --npz_fmt "$NPZ" --budgets 1500 --every 1 --seeds 42-51 --epochs $EPOCHS --out_dir $BF/mmdagg_10seed
done
for TAG in imgnet_simple imgnet_adv; do
  python vrt_forward_budget.py --tag $TAG --budget large --npz_fmt "$FEATS/$TAG/ep{ep:03d}.npz" --epochs $EPOCHS --seeds 42-51 --n_perm 2000 --out $BF/vrt_fwd/vrt_fwd_${TAG}_large.csv
done
echo ALL DONE
