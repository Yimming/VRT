#!/bin/bash
#SBATCH --job-name=c2st_raw
#SBATCH --array=0-19
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=c2st_raw_%A_%a.out
# RAW (un-normalized) feature space, Appendix A-4. C2ST (sklearn MLP, verbatim from methods.py via baselines2_stats.py) at 1,500 and 10,000 per split,
# every saved epoch, 10 seeds. 10,000 = Table 1 matched budget, 1,500 = appendix small tier.

source /home3/p318770/venvs/simclr_vrt_gpu/venv311/bin/activate
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

FEATS=/scratch/p318770/model_zoos/ladder_feats
CODE=/scratch/p318770/ssl_supcon_code
OUTDIR=/home3/p318770/venvs/overfitting/benchmark_final/c2st_raw_10seed
cd $CODE

TAGS=(r50 phase_bs8 phase_bs128 phase_bs1024 v7 ablA ablB ablC ablD wp \
      zoo_c10_m0 zoo_c10_m1 zoo_c10_m2 zoo_c10_m3 zoo_c10_m4 \
      zoo_c100_m0 zoo_c100_m1 zoo_c100_m2 zoo_c100_m3 zoo_c100_m4)
TAG=${TAGS[$SLURM_ARRAY_TASK_ID]}

python base_fill.py --stat c2st --tag $TAG --npz_fmt "$FEATS/$TAG/ep{ep:03d}.npz" \
  --budgets 1500,10000 --every 1 --raw --seeds 42-51 --epochs 0-1000 --out_dir $OUTDIR
