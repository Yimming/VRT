#!/bin/bash
# check_progress.sh — status of the running camera-ready jobs (#27102), written October 14.
# Run on the cluster after activating the venv:  bash /scratch/p318770/ssl_supcon_code_s50/check_progress.sh
# When timeouts are resubmitted, add the new job IDs to EVAL_RE or TRAIN_RE below. GPU jobs submitted after October 15
# (SimCLR MMD-on-c-hat phase 4 and any resubmission of its tasks) are read from phase4_jobid.txt in the code folder, one
# job ID per line; the submit line in the notes appends to that file with tee -a.
cd /scratch/p318770/ssl_supcon_code_s50 || exit 1
EVAL=32709214;  EVAL_RE=32767293,32768778,32775405,32783737,32798517,32832499     # eval-mode MMDAgg tail and its 24 h resubmits
TRAIN=32709215; TRAIN_RE=32767294,32768779,32775406,32783738,32798518,32832500    # train-mode MMDAgg tail and its 24 h resubmits
GPU="32785279 32784945 32816285 32818592 32822329 32832502 $(cat phase4_jobid.txt 2>/dev/null | tr '\n' ' ')" # SimCLR c100 rerun; remaining models; SimCLR phase 3; c-hat diagnostics; phase 4
GPU=$(echo $GPU)   # single spaces, no trailing space

echo "== job states (count, job, state)"
sacct -j $EVAL,$TRAIN,$EVAL_RE,$TRAIN_RE,${GPU// /,} -X -n -o JobID%25,State | awk '{split($1,a,"_"); print a[1], $2}' | sort | uniq -c

echo "== MMDAgg timeouts not yet resubmitted"
for P in "$EVAL $EVAL_RE tail_simclr_mmdagg_eval.txt" "$TRAIN $TRAIN_RE tail_simclr_mmdagg_train.txt"; do
  set -- $P
  T=$(sacct -j $1 -X -n -o JobID%25,State | awk '$2=="TIMEOUT"{split($1,a,"_"); print a[2]}' | sort)
  R=$(sacct -j $2 -X -n -o JobID%25 | awk '{split($1,a,"_"); print a[2]}' | sort -u)
  N=$(comm -23 <(echo "$T") <(echo "$R") | grep -v '^$' | paste -sd, -)
  if [ -n "$N" ]; then echo "$1: $N   resubmit with:  sbatch --time=24:00:00 --array=$N job_s50_cpu.sh $3"; else echo "$1: none"; fi
done

echo "== SimCLR MMDAgg cells (final = every checkpoint after the onset evaluated)"
python simclr_tail_check.py | grep "mmdagg 5000"

echo "== MMD on c-hat (onsets; SimCLR lines also show the capped-value check)"
python onc_summary.py --brackets | grep -v "no MMD-on-c-hat output yet"

echo "== train-mode 1,500-point fill (fill_train1500_cpu.txt)"
python fill_train1500_status.py

echo "== Python errors in the GPU and CPU logs of the jobs above (nothing listed = no error)"
for J in $GPU; do grep -l Traceback /scratch/p318770/ssl_supcon_code*/s50_logs/gpu_${J}_*.out /scratch/p318770/ssl_supcon_code*/s50_logs/cpu_${J}_*.out 2>/dev/null; done
