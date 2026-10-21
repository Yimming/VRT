#!/bin/bash
#SBATCH --job-name=null10_cpu
#SBATCH --array=0-11
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=null10_cpu_%A_%a.out
# Ten analysis seeds per random init (42-51) on the saved random-initialization features, CPU part:
#   tasks 0-3  forward VRT, both budgets, one feature set each
#   tasks 4-7  reversed VRT, both budgets, one feature set each
#   tasks 8-11 C2ST at 1,500 and 10,000, one feature set each
# Feature sets in order: imgnet_rand_eval, imgnet_rand_train, imgnet_rand_eval_trtr, imgnet_rand_train_trtr.
: "${SLURM_ARRAY_TASK_ID:?run with sbatch}"
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source /home3/p318770/venvs/simclr_vrt_gpu/venv311/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
FEATS=/scratch/p318770/model_zoos/ladder_feats
BF=/home3/p318770/venvs/overfitting/benchmark_final/s50/null10
mkdir -p $BF/vrt_fwd $BF/reversed $BF/c2st
cd /scratch/p318770/ssl_supcon_code_s50
TAGS=(imgnet_rand_eval imgnet_rand_train imgnet_rand_eval_trtr imgnet_rand_train_trtr)
i=$SLURM_ARRAY_TASK_ID; TAG=${TAGS[$((i % 4))]}; NPZ="$FEATS/$TAG/ep{ep:03d}.npz"
if [ $i -le 3 ]; then
  for B in large small; do
    python vrt_forward_budget.py  --tag $TAG --budget $B --npz_fmt "$NPZ" --epochs 0-29 --seeds 42-51 --n_perm 2000 --out $BF/vrt_fwd/vrt_fwd_${TAG}_${B}.csv
  done
elif [ $i -le 7 ]; then
  for B in large small; do
    python vrt_reversed_budget.py --tag $TAG --budget $B --npz_fmt "$NPZ" --epochs 0-29 --seeds 42-51 --n_perm 2000 --out $BF/reversed/vrt_rev_${TAG}_${B}.csv
  done
else
  python base_fill.py --stat c2st --tag $TAG --npz_fmt "$NPZ" --budgets 1500,10000 --every 1 --seeds 42-51 --epochs 0-29 --out_dir $BF/c2st
fi
rc=$?; echo "task $i ($TAG) finished rc=$rc"; exit $rc
