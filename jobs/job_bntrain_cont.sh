#!/bin/bash
#SBATCH --job-name=bntr_c
#SBATCH --partition=regular
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/p318770/ssl_supcon_code/bntr_c_%j.out
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source /home3/p318770/venvs/simclr_vrt_gpu/venv311/bin/activate
cd /scratch/p318770/ssl_supcon_code
F=/scratch/p318770/model_zoos/ladder_feats
O=/scratch/p318770/model_zoos/ladder_out
LAST=$(awk -F, 'NR>1{if($2+0>m)m=$2}END{print m}' $O/ladder_r50_bntrain.csv)
echo "r50_bntrain resuming from epoch $((LAST+1))"
python ladder_stats.py --model r50_bntrain --feats_dir $F/r50_bntrain --out_dir $O \
    --epochs $((LAST+1))-200 --mmd_full_every 5
echo "End: $(date)"
