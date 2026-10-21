#!/bin/bash
#SBATCH --job-name=trtr_trained_cpu
#SBATCH --array=0-19
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=trtr_trained_cpu_%A_%a.out
# CPU tests on the train-vs-train features of the trained ImageNet checkpoints, 10 seeds (42-51):
#   tasks  0-3  forward VRT, both budgets          tasks  4-7  reversed VRT, both budgets
#   tasks  8-11 C2ST at 1,500 and 10,000           tasks 12-15 MMDAgg at 5,000 (about 7.5 h)
#   tasks 16-19 MMDAgg at 1,500
# Feature sets in order: imgnet_simple_trtr, imgnet_adv_trtr, imgnet_simple_bntrain_trtr, imgnet_adv_bntrain_trtr.
# Submit after job_imgnet_trtr_trained.sh:  sbatch --dependency=afterok:<jobid> job_imgnet_trtr_trained_cpu.sh
: "${SLURM_ARRAY_TASK_ID:?run with sbatch}"
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source /home3/p318770/venvs/simclr_vrt_gpu/venv311/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
FEATS=/scratch/p318770/model_zoos/ladder_feats
BF=/home3/p318770/venvs/overfitting/benchmark_final/s50/trtr_trained
mkdir -p $BF/vrt_fwd $BF/reversed $BF/c2st $BF/mmdagg
cd /scratch/p318770/ssl_supcon_code_s50
TAGS=(imgnet_simple_trtr imgnet_adv_trtr imgnet_simple_bntrain_trtr imgnet_adv_bntrain_trtr)
i=$SLURM_ARRAY_TASK_ID; TAG=${TAGS[$((i % 4))]}; NPZ="$FEATS/$TAG/ep{ep:03d}.npz"
if [ $i -le 3 ]; then
  for B in large small; do
    python vrt_forward_budget.py  --tag $TAG --budget $B --npz_fmt "$NPZ" --epochs 1-300 --seeds 42-51 --n_perm 2000 --out $BF/vrt_fwd/vrt_fwd_${TAG}_${B}.csv
  done
elif [ $i -le 7 ]; then
  for B in large small; do
    python vrt_reversed_budget.py --tag $TAG --budget $B --npz_fmt "$NPZ" --epochs 1-300 --seeds 42-51 --n_perm 2000 --out $BF/reversed/vrt_rev_${TAG}_${B}.csv
  done
elif [ $i -le 11 ]; then
  python base_fill.py --stat c2st --tag $TAG --npz_fmt "$NPZ" --budgets 1500,10000 --every 1 --seeds 42-51 --epochs 1-300 --out_dir $BF/c2st
else
  python -c "import mmdagg" || { echo "mmdagg not importable"; exit 1; }
  B=$([ $i -le 15 ] && echo 5000 || echo 1500)
  python base_fill.py --stat mmdagg --tag $TAG --npz_fmt "$NPZ" --budgets $B --every 1 --seeds 42-51 --epochs 1-300 --out_dir $BF/mmdagg
fi
rc=$?; echo "task $i ($TAG) finished rc=$rc"; exit $rc
