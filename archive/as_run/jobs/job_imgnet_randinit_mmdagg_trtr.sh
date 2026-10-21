#!/bin/bash
#SBATCH --job-name=imgnet_randagg_trtr
#SBATCH --array=0-1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=imgnet_randagg_trtr_%A_%a.out
# MMDAgg at 1,500 and 5,000 on the train-vs-train random-initialization features (task 0 eval-mode, task 1
# train-mode), one analysis seed per init. Submit after job_imgnet_randinit_trtr.sh has written the 30 files per mode.
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source /home3/p318770/venvs/simclr_vrt_gpu/venv311/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python -c "from mmdagg.np import mmdagg" || { echo "mmdagg import failed"; exit 1; }
FEATS=/scratch/p318770/model_zoos/ladder_feats
BF=/home3/p318770/venvs/overfitting/benchmark_final/s50/null_trtr
cd /scratch/p318770/ssl_supcon_code_s50
MODES=(eval train); TAG=imgnet_rand_${MODES[$SLURM_ARRAY_TASK_ID]}_trtr
python base_fill.py --stat mmdagg --tag $TAG --npz_fmt "$FEATS/$TAG/ep{ep:03d}.npz" --budgets 1500 --every 1 --seeds 42 --epochs 0-29 --out_dir $BF/mmdagg
python base_fill.py --stat mmdagg --tag $TAG --npz_fmt "$FEATS/$TAG/ep{ep:03d}.npz" --budgets 5000 --every 1 --seeds 42 --epochs 0-29 --out_dir $BF/mmdagg
echo DONE
