#!/bin/bash
#SBATCH --job-name=gpurefill
#SBATCH --partition=gpushort
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --array=0-14
#SBATCH --output=/scratch/p318770/ssl_supcon_code/gpurefill_%a_%j.out
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source /home3/p318770/venvs/simclr_vrt_gpu/venv311/bin/activate
cd /scratch/p318770/ssl_supcon_code
F=/scratch/p318770/model_zoos/ladder_feats
O=/scratch/p318770/model_zoos/ladder_out
E5="5,15,25,35,45"
E5L="5,15,25,35,45,55"
U=( ablA ablB ablC ablD zoo_c10_m0 zoo_c10_m1 zoo_c10_m2 zoo_c10_m3 zoo_c10_m4 \
    zoo_c100_m0 zoo_c100_m1 zoo_c100_m2 zoo_c100_m3 zoo_c100_m4 r50_c3 )
EP=( $E5 $E5 $E5 $E5 $E5 $E5 $E5 $E5 $E5 $E5L $E5L $E5L $E5L $E5L 199 )
i=$SLURM_ARRAY_TASK_ID
FE=$(echo "${U[$i]}" | sed -e 's/_c[0-9]*$//')
python gpufill2.py --unit ${U[$i]} --feats_dir $F/$FE \
    --epochs "${EP[$i]}" --out_dir $O/bigfill2_${U[$i]}
