#!/bin/bash
# Fill the gap: ladder for v7 and ablA (r50 ladder is job 30259562).
#SBATCH --job-name=ladder_v7A
#SBATCH --partition=regular
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/p318770/ssl_supcon_code/ladder_v7A_%j.out

module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source /home3/p318770/venvs/simclr_vrt_gpu/venv311/bin/activate

FEATS=/scratch/p318770/model_zoos/ladder_feats
OUT=/scratch/p318770/model_zoos/ladder_out

cd /scratch/p318770/ssl_supcon_code
echo "Start: $(date)"
python ladder_stats.py --model v7   --feats_dir $FEATS/v7   --out_dir $OUT/ladder_v7
python ladder_stats.py --model ablA --feats_dir $FEATS/ablA --out_dir $OUT/ladder_ablA
echo "End: $(date)"
