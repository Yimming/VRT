#!/bin/bash
#SBATCH --job-name=ksweep15
#SBATCH --partition=regular
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --array=0-22
#SBATCH --output=/scratch/p318770/ssl_supcon_code/ksweep15_%a_%j.out
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source /home3/p318770/venvs/simclr_vrt_gpu/venv311/bin/activate
cd /scratch/p318770/ssl_supcon_code
F=/scratch/p318770/model_zoos/ladder_feats
O=/scratch/p318770/model_zoos/ladder_out
U=(  r50 r50 r50 r50 v7 v7 v7 v7 ablA ablB ablC ablD \
     zoo_c10_m0 zoo_c10_m1 zoo_c10_m2 zoo_c10_m3 zoo_c10_m4 \
     zoo_c100_m0 zoo_c100_m1 zoo_c100_m2 zoo_c100_m3 zoo_c100_m4 wp )
LO=( 0 51 101 151 0 50 100 150 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 )
HI=( 50 100 150 200 49 99 149 199 49 49 49 49  50 50 50 50 50  60 60 60 60 60  1000 )
i=$SLURM_ARRAY_TASK_ID
python ksweep15_fill.py --unit ${U[$i]} --feats_dir $F/${U[$i]} \
    --lo ${LO[$i]} --hi ${HI[$i]} --out_dir $O
