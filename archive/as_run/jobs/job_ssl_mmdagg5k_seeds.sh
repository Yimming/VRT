#!/bin/bash
#SBATCH --job-name=sslagg50
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --output=s50_logs/sslagg50_%A_%a.out
# SimCLR MMDAgg 5,000: seeds 52-91 on every epoch of the existing coarse-to-fine grid (one epoch per array task),
# so that the SimCLR MMDAgg cells of Table 1 are 50 seeds like every other cell. Task file: one "dataset epoch" per
# line, made by ssl_agg_tasks.sh. Usage:  sbatch --array=0-$((N-1)) job_ssl_mmdagg5k_seeds.sh ssl_agg_tasks.txt
: "${SLURM_ARRAY_TASK_ID:?run with sbatch --array}"
TASKS=${1:?task file}
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TASKS"); set -- $LINE
DS=$1; EP=$2
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source /home3/p318770/venvs/simclr_vrt_gpu/venv311/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd /scratch/p318770/ssl_supcon_code_s50
python -c "import mmdagg" || { echo "mmdagg not importable"; exit 1; }
CK=$(ls -d /scratch/p318770/model_zoos/ssl_supcon/SimCLR_${DS}_* | head -1)
echo "task $SLURM_ARRAY_TASK_ID: $DS epoch $EP  ckpt dir $CK"
python ssl_mmdagg5k_seeds.py --ckpt_dir "$CK" --dataset $DS --epochs $EP --seeds 52-91
rc=$?; echo "task $SLURM_ARRAY_TASK_ID ($DS $EP) finished rc=$rc"; exit $rc
