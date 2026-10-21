#!/bin/bash
#SBATCH --job-name=mmd_p2000
#SBATCH --array=0-43
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=mmd_p2000_%A_%a.out
# MMD (Gaussian RBF, median heuristic, unbiased MMD^2), camera-ready unified protocol: 2,000 permutations, 10 seeds (42-51), every saved epoch,
# two per-split budgets (1,500 = Table 1 and the small matched budget; 10,000 = large matched budget).
# mmd_tst_fast.py is verified identical to methods_v2.mmd_test (same seed -> same p).
# task i: budget = BUDGETS[i / 22], tag = TAGS[i % 22].

source /home3/p318770/venvs/simclr_vrt_gpu/venv311/bin/activate
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

FEATS=/scratch/p318770/model_zoos/ladder_feats
SSL=/scratch/p318770/model_zoos/ssl_supcon_feats
CODE=/scratch/p318770/ssl_supcon_code
OUTDIR=/home3/p318770/venvs/overfitting/benchmark_final/mmd_p2000
mkdir -p $OUTDIR
cd $CODE

TAGS=(r50 phase_bs8 phase_bs128 phase_bs1024 v7 ablA ablB ablC ablD wp \
      zoo_c10_m0 zoo_c10_m1 zoo_c10_m2 zoo_c10_m3 zoo_c10_m4 \
      zoo_c100_m0 zoo_c100_m1 zoo_c100_m2 zoo_c100_m3 zoo_c100_m4 \
      simclr_c10 simclr_c100)
BUDGETS=(1500 10000)
TAG=${TAGS[$((SLURM_ARRAY_TASK_ID % 22))]}
N=${BUDGETS[$((SLURM_ARRAY_TASK_ID / 22))]}

case $TAG in
  simclr_c10)  DIR=$SSL/feats_cifar10_SimCLR_cifar10_resnet50_lr_0.25_decay_0.0001_bsz_512_temp_0.5_trial_ssl_c10_s0_cosine_warm ;;
  simclr_c100) DIR=$SSL/feats_cifar100_SimCLR_cifar100_resnet50_lr_0.25_decay_0.0001_bsz_512_temp_0.5_trial_ssl_c100_s0_cosine_warm ;;
  *)           DIR=$FEATS/$TAG ;;
esac
[ -d "$DIR" ] || { echo "no feature folder: $DIR"; exit 1; }

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then python mmd_tst_fast.py --selftest || exit 1; fi

python mmd_tst_fast.py --tag $TAG --npz_fmt "$DIR/ep{ep:03d}.npz" \
  --n_samples $N --n_perm 2000 --seeds 42-51 --epochs 0-1000 --backend torch \
  --out $OUTDIR/mmd_${TAG}_n${N}.csv
