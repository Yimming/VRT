#!/bin/bash
# null50_make_tasks.sh — task files that bring every null experiment to 50 analysis seeds (42-91):
#   random-initialization features  -> s50/null10        (4 feature sets x 30 inits, "epochs" 0-29)
#   trained checkpoints, train vs further train -> s50/trtr_trained (4 feature sets x 9 checkpoints)
# Seeds already present are kept and only the missing ones are computed:
#   VRT forward, VRT reversed, MMD, ED, Henze/Schilling: seeds 52-91 are appended to the existing CSV files
#     (vrt_forward resumes per (epoch, seed); the others count rows per epoch, 10 present < 40 requested)
#   C2ST and MMDAgg: seeds 52-91 go to new CSV files with an _s52-91 suffix, which the harvests pick up by glob
#   MMDAgg at 1,500 on the random-init sets has only seed 42 so far: seeds 43-91 (or 52-91 if the 10-seed job
#     job_null_mmdagg1500_seeds.sh already wrote seeds 43-51; detected automatically)
# Every task writes its own files, so all tasks can run at the same time. Usage, on the cluster:
#   cd /scratch/p318770/ssl_supcon_code_s50 && bash null50_make_tasks.sh
# then the two sbatch lines it prints (the s50 drivers job_s50_gpu.sh and job_s50_cpu.sh run one line per task).
set -e
CODE=${CODE:-/scratch/p318770/ssl_supcon_code_s50}
FEATS=${FEATS:-/scratch/p318770/model_zoos/ladder_feats}
S50=${S50:-/home3/p318770/venvs/overfitting/benchmark_final/s50}
cd "$CODE"
NEW=52-91
if ls "$S50"/null10/mmdagg/mmdagg_*_b1500_s43-51_* >/dev/null 2>&1; then SEEDS15=52-91; else SEEDS15=43-91; fi
STAMP=$(date +%Y%m%d_%H%M)
G=null50_gpu_tasks_$STAMP.txt; C=null50_cpu_tasks_$STAMP.txt
: > "$G"; : > "$C"
RAND="imgnet_rand_eval imgnet_rand_train imgnet_rand_eval_trtr imgnet_rand_train_trtr"
TRAINED="imgnet_simple_trtr imgnet_adv_trtr imgnet_simple_bntrain_trtr imgnet_adv_bntrain_trtr"

# missing feature files would make every line of that set a silent no-op: check first
for TAG in $RAND; do n=$(ls "$FEATS/$TAG"/ep*.npz 2>/dev/null | wc -l); [ "$n" -eq 30 ] || { echo "ERROR: $FEATS/$TAG has $n of 30 files"; exit 1; }; done
for TAG in $TRAINED; do n=$(ls "$FEATS/$TAG"/ep*.npz 2>/dev/null | wc -l); [ "$n" -eq 9 ] || { echo "ERROR: $FEATS/$TAG has $n of 9 files"; exit 1; }; done

for TAG in $RAND $TRAINED; do
  case " $RAND " in *" $TAG "*) BF=$S50/null10; EPS=0-29;; *) BF=$S50/trtr_trained; EPS=1-300;; esac
  NPZ="'$FEATS/$TAG/ep{ep:03d}.npz'"
  # GPU, one task per feature set: MMD, ED, Henze/Schilling at 1,500 and 10,000
  L=""
  for N in 1500 10000; do
    L="$L${L:+ && }python mmd_tst_fast.py --tag $TAG --npz_fmt $NPZ --n_samples $N --n_perm 2000 --seeds $NEW --epochs $EPS --backend torch --out $BF/mmd_p2000/mmd_${TAG}_n${N}.csv"
    L="$L && python energy_tst_fast.py --tag $TAG --npz_fmt $NPZ --n_samples $N --n_perm 2000 --seeds $NEW --epochs $EPS --backend torch --out $BF/ed_p2000/ed_${TAG}_n${N}.csv"
    L="$L && python knn_tst_budget.py --tag $TAG --npz_fmt $NPZ --n_samples $N --k 3 --n_perm 2000 --seeds $NEW --epochs $EPS --backend torch --out $BF/knn_v2/knn_${TAG}_n${N}_part.csv"
  done
  echo "$L" >> "$G"
  # CPU, three tasks per feature set: VRT forward (both budgets), VRT reversed (both budgets), C2ST (both budgets)
  echo "python vrt_forward_budget.py --tag $TAG --budget large --npz_fmt $NPZ --epochs $EPS --seeds $NEW --n_perm 2000 --out $BF/vrt_fwd/vrt_fwd_${TAG}_large.csv && python vrt_forward_budget.py --tag $TAG --budget small --npz_fmt $NPZ --epochs $EPS --seeds $NEW --n_perm 2000 --out $BF/vrt_fwd/vrt_fwd_${TAG}_small.csv" >> "$C"
  echo "python vrt_reversed_budget.py --tag $TAG --budget large --npz_fmt $NPZ --epochs $EPS --seeds $NEW --n_perm 2000 --out $BF/reversed/vrt_rev_${TAG}_large.csv && python vrt_reversed_budget.py --tag $TAG --budget small --npz_fmt $NPZ --epochs $EPS --seeds $NEW --n_perm 2000 --out $BF/reversed/vrt_rev_${TAG}_small.csv" >> "$C"
  echo "python base_fill.py --stat c2st --tag $TAG --npz_fmt $NPZ --budgets 1500,10000 --every 1 --seeds $NEW --epochs $EPS --out_dir $BF/c2st --out_suffix _s$NEW" >> "$C"
done

# MMDAgg at 5,000: one task per (set, init) and per (set, checkpoint), 40 seeds each, about 3 to 6 h
for TAG in $RAND; do for I in $(seq 0 29); do
  echo "python base_fill.py --stat mmdagg --tag $TAG --npz_fmt '$FEATS/$TAG/ep{ep:03d}.npz' --budgets 5000 --every 1 --seeds $NEW --epochs $I --out_dir $S50/null10/mmdagg --out_suffix _s${NEW}_e$I" >> "$C"
done; done
for TAG in $TRAINED; do for EP in 1 5 20 50 100 150 200 250 300; do
  echo "python base_fill.py --stat mmdagg --tag $TAG --npz_fmt '$FEATS/$TAG/ep{ep:03d}.npz' --budgets 5000 --every 1 --seeds $NEW --epochs $EP --out_dir $S50/trtr_trained/mmdagg --out_suffix _s${NEW}_e$EP" >> "$C"
done; done
# MMDAgg at 1,500: random-init sets 5 inits per task; trained sets one task per set
for TAG in $RAND; do for LO in 0 5 10 15 20 25; do HI=$((LO + 4))
  echo "python base_fill.py --stat mmdagg --tag $TAG --npz_fmt '$FEATS/$TAG/ep{ep:03d}.npz' --budgets 1500 --every 1 --seeds $SEEDS15 --epochs $LO-$HI --out_dir $S50/null10/mmdagg --out_suffix _s${SEEDS15}_c$LO-$HI" >> "$C"
done; done
for TAG in $TRAINED; do
  echo "python base_fill.py --stat mmdagg --tag $TAG --npz_fmt '$FEATS/$TAG/ep{ep:03d}.npz' --budgets 1500 --every 1 --seeds $NEW --epochs 1-300 --out_dir $S50/trtr_trained/mmdagg --out_suffix _s$NEW" >> "$C"
done

NG=$(wc -l < "$G"); NC=$(wc -l < "$C")
echo "GPU tasks: $NG -> $G"
echo "CPU tasks: $NC -> $C   (MMDAgg 1,500 random-init seeds: $SEEDS15)"
echo "submit:"
echo "  sbatch --array=0-$((NG - 1)) job_s50_gpu.sh $G"
echo "  sbatch --array=0-$((NC - 1))%120 job_s50_cpu.sh $C"
