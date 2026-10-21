#!/bin/bash
#SBATCH --job-name=s50cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/p318770/ssl_supcon_code_s50/s50_logs/cpu_%A_%a.out
# 50-seed rerun, CPU family (VRT, C2ST, MMDAgg, SimCLR MMDAgg). Usage:
#   sbatch --array=0-<N-1>%120 job_s50_cpu.sh <task file>
# Same conventions as job_s50_gpu.sh. module FIRST, then venv (mmdagg lives in the venv only).
: "${SLURM_ARRAY_TASK_ID:?submit with --array}"
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source /home3/p318770/venvs/simclr_vrt_gpu/venv311/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd /scratch/p318770/ssl_supcon_code_s50
TASKS=${1:?task file required}
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TASKS")
[ -n "$LINE" ] || { echo "no task line $((SLURM_ARRAY_TASK_ID + 1)) in $TASKS"; exit 1; }
case "$LINE" in *mmdagg*|*"--group slow"*) python -c "from mmdagg.np import mmdagg" || { echo "mmdagg import failed"; exit 1; } ;; esac
echo "task $SLURM_ARRAY_TASK_ID on $(hostname): $LINE"
eval "$LINE"; rc=$?
echo "task $SLURM_ARRAY_TASK_ID finished rc=$rc"
exit $rc
