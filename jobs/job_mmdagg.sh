#!/bin/bash
#SBATCH --job-name=mmdagg
#SBATCH --array=0-39
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=mmdagg_%A_%a.out
# MMDAgg (mmdagg.np, B1 = B2 = 2,000, verbatim protocol) with 10 seeds.
# tasks 0-19 : 1,500 per split, every saved epoch      (appendix small tier, also the submission's protocol)
# tasks 20-39: 5,000 per split, every 5th epoch        (Table 1 matched tier; 10,000 exceeds the numpy
#              implementation's memory, 5,000 is the largest tractable size, footnoted in the paper)

# module FIRST, then venv, so that `python` is the venv interpreter (mmdagg is installed in the venv only)
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source /home3/p318770/venvs/simclr_vrt_gpu/venv311/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python -c "from mmdagg.np import mmdagg" || { echo "mmdagg import failed"; exit 1; }

FEATS=/scratch/p318770/model_zoos/ladder_feats
CODE=/scratch/p318770/ssl_supcon_code
OUTDIR=/home3/p318770/venvs/overfitting/benchmark_final/mmdagg_10seed
cd $CODE

TAGS=(r50 phase_bs8 phase_bs128 phase_bs1024 v7 ablA ablB ablC ablD wp \
      zoo_c10_m0 zoo_c10_m1 zoo_c10_m2 zoo_c10_m3 zoo_c10_m4 \
      zoo_c100_m0 zoo_c100_m1 zoo_c100_m2 zoo_c100_m3 zoo_c100_m4)
TAG=${TAGS[$((SLURM_ARRAY_TASK_ID % 20))]}
if [ $SLURM_ARRAY_TASK_ID -lt 20 ]; then B=1500; EVERY=1; else B=5000; EVERY=5; fi

python base_fill.py --stat mmdagg --tag $TAG --npz_fmt "$FEATS/$TAG/ep{ep:03d}.npz" \
  --budgets $B --every $EVERY --seeds 42-51 --epochs 0-1000 --out_dir $OUTDIR
