#!/bin/bash
#SBATCH --job-name=bntrain_base
#SBATCH --array=0-104
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=bntrain_base_%A_%a.out
# Section 4.4 check on the 15 BatchNorm CIFAR settings whose train-mode-BatchNorm features already exist
# (rebuttal re-extraction, <setting>_bntrain/ep{ep:03d}.npz): the baselines on those features, unified
# protocol (2,000 permutations, 10 seeds, every saved epoch, both budgets), so that each eval-mode onset
# gets its train-mode counterpart. MMD and ED in both the L2-normalized and the raw feature space
# (the raw-space gains of Appendix A-4 are the ones suspected to ride on the BatchNorm norm channel),
# Schilling/Henze normalized, forward VRT at both budgets (same scripts as the ImageNet check).
# Tasks 0-89: setting = i % 15, budget = BUDGETS[(i / 15) % 2], group = {mmd, ed, knn}[i / 30].
# Tasks 90-104: forward VRT (large via vrt_forward_budget.py, small via ksweep15_fill.py) for setting i - 90.
# Results go to benchmark_final/bntrain/<same subdirectory names as the eval-mode results>; harvest_bntrain.py.
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source /home3/p318770/venvs/simclr_vrt_gpu/venv311/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
FEATS=/scratch/p318770/model_zoos/ladder_feats
BF=/home3/p318770/venvs/overfitting/benchmark_final/bntrain
mkdir -p $BF/mmd_p2000 $BF/mmd_raw_p2000 $BF/ed_p2000 $BF/ed_raw_p2000 $BF/knn_v2 $BF/vrt_fwd $BF/ksweep15
cd /scratch/p318770/ssl_supcon_code
SETTINGS=(r50 ablA ablB ablC ablD zoo_c10_m0 zoo_c10_m1 zoo_c10_m2 zoo_c10_m3 zoo_c10_m4 zoo_c100_m0 zoo_c100_m1 zoo_c100_m2 zoo_c100_m3 zoo_c100_m4)
BUDGETS=(1500 10000)
TASKGROUPS=(mmd ed knn)
i=$SLURM_ARRAY_TASK_ID
if [ $i -ge 90 ]; then
  S=${SETTINGS[$((i - 90))]}; TAG=${S}_bntrain; NPZ="$FEATS/$TAG/ep{ep:03d}.npz"
  echo "task $i: forward VRT on $TAG ($(ls $FEATS/$TAG/ep*.npz | wc -l) epochs)"
  python vrt_forward_budget.py --tag $TAG --budget large --npz_fmt "$NPZ" --epochs 0-1000 --seeds 42-51 --n_perm 2000 --out $BF/vrt_fwd/vrt_fwd_${TAG}_large.csv
  python ksweep15_fill.py --unit $TAG --feats_dir $FEATS/$TAG --out_dir $BF/ksweep15 --lo 0 --hi 1000
else
  S=${SETTINGS[$((i % 15))]}; N=${BUDGETS[$(((i / 15) % 2))]}; G=${TASKGROUPS[$((i / 30))]}
  TAG=${S}_bntrain; NPZ="$FEATS/$TAG/ep{ep:03d}.npz"
  echo "task $i: $G on $TAG at n=$N ($(ls $FEATS/$TAG/ep*.npz | wc -l) epochs)"
  case $G in
    mmd)
      python mmd_tst_fast.py --tag $TAG --npz_fmt "$NPZ" --n_samples $N --n_perm 2000 --seeds 42-51 --epochs 0-1000 --backend torch       --out $BF/mmd_p2000/mmd_${TAG}_n${N}.csv
      python mmd_tst_fast.py --tag $TAG --npz_fmt "$NPZ" --n_samples $N --n_perm 2000 --seeds 42-51 --epochs 0-1000 --backend torch --raw --out $BF/mmd_raw_p2000/mmd_${TAG}_n${N}.csv ;;
    ed)
      python energy_tst_fast.py --tag $TAG --npz_fmt "$NPZ" --n_samples $N --n_perm 2000 --seeds 42-51 --epochs 0-1000 --backend torch       --out $BF/ed_p2000/ed_${TAG}_n${N}.csv
      python energy_tst_fast.py --tag $TAG --npz_fmt "$NPZ" --n_samples $N --n_perm 2000 --seeds 42-51 --epochs 0-1000 --backend torch --raw --out $BF/ed_raw_p2000/ed_${TAG}_n${N}.csv ;;
    knn)
      python knn_tst_budget.py --tag $TAG --npz_fmt "$NPZ" --n_samples $N --k 3 --n_perm 2000 --seeds 42-51 --epochs 0-1000 --backend torch --out $BF/knn_v2/knn_${TAG}_n${N}_part${TAG}.csv ;;
  esac
fi
echo DONE
