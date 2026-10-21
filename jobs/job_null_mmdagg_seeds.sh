#!/bin/bash
#SBATCH --job-name=null10_agg
#SBATCH --array=0-11
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=09:00:00
#SBATCH --output=null10_agg_%A_%a.out
# MMDAgg at 5,000 per split on the random-initialization features, analysis seeds 43-51 (seed 42 exists from the
# one-seed runs, copied into s50/null10/mmdagg by job_null_seeds_gpu.sh), so that every cell of the two null tables
# has ten seeds per init. 4 feature sets x 3 chunks of 10 inits; about 7.5 h per task.
#   sets: imgnet_rand_eval, imgnet_rand_train (train vs validation), imgnet_rand_eval_trtr, imgnet_rand_train_trtr
: "${SLURM_ARRAY_TASK_ID:?run with sbatch}"
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source /home3/p318770/venvs/simclr_vrt_gpu/venv311/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python -c "import mmdagg" || { echo "mmdagg not importable"; exit 1; }
FEATS=/scratch/p318770/model_zoos/ladder_feats
BF=/home3/p318770/venvs/overfitting/benchmark_final/s50/null10
mkdir -p $BF/mmdagg
cd /scratch/p318770/ssl_supcon_code_s50
TAGS=(imgnet_rand_eval imgnet_rand_train imgnet_rand_eval_trtr imgnet_rand_train_trtr)
i=$SLURM_ARRAY_TASK_ID; TAG=${TAGS[$((i % 4))]}; C=$((i / 4)); LO=$((C * 10)); HI=$((C * 10 + 9))
echo "task $i: $TAG inits $LO-$HI seeds 43-51"
python base_fill.py --stat mmdagg --tag $TAG --npz_fmt "$FEATS/$TAG/ep{ep:03d}.npz" --budgets 5000 --every 1 --seeds 43-51 --epochs $LO-$HI --out_dir $BF/mmdagg --out_suffix _s43-51_c$LO-$HI
rc=$?; echo "task $i ($TAG $LO-$HI) finished rc=$rc"; exit $rc
