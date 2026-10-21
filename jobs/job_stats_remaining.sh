#!/bin/bash
# ONE array: ladder + unnorm + baselines2 for ALL remaining models.
# Per-model big-budget intervals account for checkpoint spacing
# (wp checkpoints are every 10 epochs -> use 50; phase eps 100/125/150 -> 5).
#SBATCH --job-name=stats_rem
#SBATCH --partition=regular
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/p318770/ssl_supcon_code/stats_rem_%a_%j.out
#SBATCH --array=0-16

module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source /home3/p318770/venvs/simclr_vrt_gpu/venv311/bin/activate

FEATS=/scratch/p318770/model_zoos/ladder_feats
OUT=/scratch/p318770/model_zoos/ladder_out

MODELS=(ablB ablC ablD \
        wp phase_bs8 phase_bs128 phase_bs1024 \
        zoo_c10_m0 zoo_c10_m1 zoo_c10_m2 zoo_c10_m3 zoo_c10_m4 \
        zoo_c100_m0 zoo_c100_m1 zoo_c100_m2 zoo_c100_m3 zoo_c100_m4)
M=${MODELS[$SLURM_ARRAY_TASK_ID]}

# big-budget epoch interval per model family
case $M in
  wp)        EV=50 ;;
  phase_*)   EV=5 ;;
  *)         EV=2 ;;
esac

cd /scratch/p318770/ssl_supcon_code
echo "Start: $(date)  model=$M  big_every=$EV"
python ladder_stats.py     --model $M --feats_dir $FEATS/$M --out_dir $OUT/ladder_$M
python unnorm_stats.py     --model $M --feats_dir $FEATS/$M --out_dir $OUT/unnorm_$M --mmd10k_every $EV
python baselines2_stats.py --model $M --feats_dir $FEATS/$M --out_dir $OUT/base2_$M  --big_every $EV --c2st_paper
echo "End: $(date)"
