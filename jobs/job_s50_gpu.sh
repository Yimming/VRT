#!/bin/bash
#SBATCH --job-name=s50gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/p318770/ssl_supcon_code_s50/s50_logs/gpu_%A_%a.out
# 50-seed rerun, GPU family. Usage (array range printed by s50_make_tasks.py):
#   sbatch --array=0-<N-1>%60 job_s50_gpu.sh <task file>
# Task i executes line i+1 of the task file. Every script resumes, so a failed or timed-out task is
# resubmitted with the same index and the same task file:  sbatch --array=<i> job_s50_gpu.sh <task file>
# Never resubmit an index that is still running (two processes would append to the same CSV).
: "${SLURM_ARRAY_TASK_ID:?submit with --array}"
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source /home3/p318770/venvs/simclr_vrt_gpu/venv311/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd /scratch/p318770/ssl_supcon_code_s50
TASKS=${1:?task file required}
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TASKS")
[ -n "$LINE" ] || { echo "no task line $((SLURM_ARRAY_TASK_ID + 1)) in $TASKS"; exit 1; }
echo "task $SLURM_ARRAY_TASK_ID on $(hostname) ($(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)): $LINE"
eval "$LINE"; rc=$?
echo "task $SLURM_ARRAY_TASK_ID finished rc=$rc"
exit $rc
