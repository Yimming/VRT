#!/bin/bash
#SBATCH --job-name=refill
#SBATCH --partition=regular
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --array=0-48
#SBATCH --output=/scratch/p318770/ssl_supcon_code/refill_%a_%j.out
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source /home3/p318770/venvs/simclr_vrt_gpu/venv311/bin/activate
cd /scratch/p318770/ssl_supcon_code
F=/scratch/p318770/model_zoos/ladder_feats
O=/scratch/p318770/model_zoos/ladder_out
E5="5,15,25,35,45"
E5L="5,15,25,35,45,55"
M=( c2st c2st c2st c2st c2st c2st c2st c2st c2st c2st c2st c2st c2st c2st c2st \
    ed ed ed ed ed ed ed ed ed ed ed ed ed ed ed \
    mmdagg mmdagg mmdagg mmdagg mmdagg mmdagg mmdagg mmdagg mmdagg \
    mmdagg mmdagg mmdagg mmdagg mmdagg mmdagg mmdagg mmdagg mmdagg mmdagg )
U=( ablA ablB ablC ablD zoo_c10_m0 zoo_c10_m1 zoo_c10_m2 zoo_c10_m3 zoo_c10_m4 \
    zoo_c100_m0 zoo_c100_m1 zoo_c100_m2 zoo_c100_m3 zoo_c100_m4 r50_c3 \
    ablA ablB ablC ablD zoo_c10_m0 zoo_c10_m1 zoo_c10_m2 zoo_c10_m3 zoo_c10_m4 \
    zoo_c100_m0 zoo_c100_m1 zoo_c100_m2 zoo_c100_m3 zoo_c100_m4 r50_c3 \
    ablA ablB ablC ablD zoo_c10_m0 zoo_c10_m1 zoo_c10_m2 zoo_c10_m3 zoo_c10_m4 \
    zoo_c100_m0 zoo_c100_m1 zoo_c100_m2 zoo_c100_m3 zoo_c100_m4 \
    r50_c0 r50_c1 r50_c2 r50_c3 v7a )
EP=( $E5 $E5 $E5 $E5 $E5 $E5 $E5 $E5 $E5 $E5L $E5L $E5L $E5L $E5L 199 \
     $E5 $E5 $E5 $E5 $E5 $E5 $E5 $E5 $E5 $E5L $E5L $E5L $E5L $E5L 199 \
     $E5 $E5 $E5 $E5 $E5 $E5 $E5 $E5 $E5 \
     $E5L $E5L "5,15,25,35,45,53,55,57,59" $E5L $E5L \
     "34,36,37,38,39,41,42,43,44,46,47,48,49" \
     "91,92,93,94,96,97,98,99" \
     "138,139,141,142,143,144,146,147,148,149" \
     "188,189,191,192,193,194,196,197,198,199" \
     "79,81,82,83,84,86,87,88,89,91,92,93,94,96,97,98,99" )
i=$SLURM_ARRAY_TASK_ID
FE=$(echo "${U[$i]}" | sed -e 's/_c[0-9]*$//' -e 's/^v7[ab]$/v7/')
python bigfill2.py --method ${M[$i]} --unit ${U[$i]} --feats_dir $F/$FE \
    --only "${EP[$i]}" --out_dir $O/bigfill2_${U[$i]}
