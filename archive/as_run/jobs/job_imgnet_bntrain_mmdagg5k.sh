#!/bin/bash
#SBATCH --job-name=imgnet_bnagg
#SBATCH --array=0-5
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --output=imgnet_bnagg_%A_%a.out
# MMDAgg at 5,000 per split (matched tier) on the train-mode-BatchNorm ImageNet features, 10 seeds.
# About 50 min per checkpoint, so the 9 checkpoints of each scheme are split over 3 tasks that write
# separate files (--out_suffix); harvest_bn.py reads them together.
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source /home3/p318770/venvs/simclr_vrt_gpu/venv311/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
FEATS=/scratch/p318770/model_zoos/ladder_feats
BF=/home3/p318770/venvs/overfitting/benchmark_final
cd /scratch/p318770/ssl_supcon_code
python -c "import mmdagg" || { echo "mmdagg not importable, abort"; exit 1; }
TAGS=(imgnet_simple_bntrain imgnet_adv_bntrain)
EPGROUPS=("1,5,20" "50,100,150" "200,250,300")
TAG=${TAGS[$((SLURM_ARRAY_TASK_ID / 3))]}
G=$((SLURM_ARRAY_TASK_ID % 3))
EPS=${EPGROUPS[$G]}
echo "task $SLURM_ARRAY_TASK_ID: $TAG epochs $EPS"
python base_fill.py --stat mmdagg --tag $TAG --npz_fmt "$FEATS/$TAG/ep{ep:03d}.npz" --budgets 5000 --every 1 --seeds 42-51 --epochs $EPS --out_dir $BF/mmdagg_10seed --out_suffix _g$G
echo DONE
