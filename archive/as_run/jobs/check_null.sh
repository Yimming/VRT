#!/bin/bash
# check_null.sh — one command for every running check: the queue, then each harvest of the CIFAR null runs and the SimCLR
# MMDAgg 1,500 run. Every harvest reads whatever is finished, so it can be run at any time.
# Usage, on the cluster, after module load and source venv:
#   cd /scratch/p318770/ssl_supcon_code_s50 && bash check_null.sh            (add "pixels" to also run cifar_split_pixels.py)
cd /scratch/p318770/ssl_supcon_code_s50 || exit 1
python -c "import numpy, scipy" 2>/dev/null || {
  echo "load the environment first:"
  echo "  module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1"
  echo "  source /home3/p318770/venvs/simclr_vrt_gpu/venv311/bin/activate"
  exit 1; }
echo "=== queue: tasks per job and state"
squeue -h -r -u "$USER" -o "%F %T" | sort | uniq -c
echo "=== queue: job -> task file"
squeue -u "$USER" -h -o "%F %o" | sort -u -k1,1 | awk '{print "  " $1, $NF}'
echo
echo "=== 1. VRT on random networks, Type-I (resplit) and official split, both budgets"
python harvest_cifar_null_vrt.py
echo
echo "=== 2. every method on random networks, 10,000 per split (empty rows = that job has not produced results yet)"
python harvest_cifar_null.py
echo
echo "=== 3. blank control: the paper's rule (official split, seeds 42-91, median p) on untrained networks"
python harvest_cifar_null_rule.py
echo
echo "=== 4. SimCLR MMDAgg 1,500, every epoch (onset over the epochs finished so far)"
for ds in cifar10 cifar100; do echo "  $ds"; python ssl_peek.py $ds | grep -A1 "^mmdagg"; done
if [ "$1" = pixels ]; then
  echo
  echo "=== 5. pixel statistics, official test vs training images (about a minute)"
  python cifar_split_pixels.py
fi
