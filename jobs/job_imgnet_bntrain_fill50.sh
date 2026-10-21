#!/bin/bash
#SBATCH --job-name=imgbn50
#SBATCH --array=0-21
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=s50_logs/imgbn50_%A_%a.out
# ImageNet train-mode BatchNorm features, the two tests that the s50 generator runs on eval-mode features only,
# brought to the 50-seed protocol so that the ImageNet eval -> train table is 50 seeds in every cell:
#   tasks  0-17 : MMDAgg at 5,000 per split, one (scheme, checkpoint) per task   (about 4.5 h each)
#   tasks 18-19 : MMDAgg at 1,500 per split, all nine checkpoints of one scheme  (about 4 h each)
#   tasks 20-21 : C2ST at 1,500 and 10,000 per split, all nine checkpoints of one scheme (under 1 h)
# Output goes to the s50 tree (mmdagg/, c2st/), where harvest_s50.py reads it. Resume: complete (epoch, budget) cells are skipped.
: "${SLURM_ARRAY_TASK_ID:?run with sbatch}"
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source /home3/p318770/venvs/simclr_vrt_gpu/venv311/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd /scratch/p318770/ssl_supcon_code_s50
python -c "import mmdagg" || { echo "mmdagg not importable"; exit 1; }
F=/scratch/p318770/model_zoos/ladder_feats
BF=/home3/p318770/venvs/overfitting/benchmark_final/s50
EPS=(1 5 20 50 100 150 200 250 300)
SCHEMES=(simple adv)
i=$SLURM_ARRAY_TASK_ID
if [ $i -le 17 ]; then
  TAG=imgnet_${SCHEMES[$((i / 9))]}_bntrain; EP=${EPS[$((i % 9))]}
  python base_fill.py --stat mmdagg --tag $TAG --npz_fmt "$F/$TAG/ep{ep:03d}.npz" --budgets 5000 --every 1 --seeds 42-91 --epochs $EP --out_dir $BF/mmdagg --out_suffix _c$EP
elif [ $i -le 19 ]; then
  TAG=imgnet_${SCHEMES[$((i - 18))]}_bntrain
  python base_fill.py --stat mmdagg --tag $TAG --npz_fmt "$F/$TAG/ep{ep:03d}.npz" --budgets 1500 --every 1 --seeds 42-91 --epochs 1-300 --out_dir $BF/mmdagg --out_suffix _c1-300
else
  TAG=imgnet_${SCHEMES[$((i - 20))]}_bntrain
  python base_fill.py --stat c2st --tag $TAG --npz_fmt "$F/$TAG/ep{ep:03d}.npz" --budgets 1500,10000 --every 1 --seeds 42-91 --epochs 1-300 --out_dir $BF/c2st --out_suffix _c1-300
fi
rc=$?; echo "task $i ($TAG) finished rc=$rc"; exit $rc
