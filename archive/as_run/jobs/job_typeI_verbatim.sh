#!/bin/bash
#SBATCH --job-name=typeI_verb
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=typeI_verb_%A_%a.out
# The published Appendix C script (VRT only, eval-mode BatchNorm, random DeepResNet50, fixed 50,000 train vs 10,000 val
# images, 30 seeds here), separating the two things that differed between the Habrok run behind the 18/500 and the
# Table 1 feature pipeline:
#   task 0  cluster preprocessing (Resize 256 + ImageNet mean/std) on the partial copy /scratch/p318770/imagenet
#           (799,751 training images): the configuration that actually produced the 18/500
#   task 1  cluster preprocessing on the full ImageNet copy (1,281,167 training images)
#   task 2  paper preprocessing (Resize 224, no normalization) on the full copy: the Table 1 pipeline, i.e. the
#           configuration of the s50 null, where eval-mode VRT forward rejected 22/30
: "${SLURM_ARRAY_TASK_ID:?run with sbatch}"
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source /home3/p318770/venvs/simclr_vrt_gpu/venv311/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
BF=/home3/p318770/venvs/overfitting/benchmark_final/s50/null
cd /scratch/p318770/ssl_supcon_code_s50
if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
  python -W ignore -u typeI_imagenet_verbatim.py --preproc cluster --data_root /scratch/p318770/imagenet --n_seeds 30 --out $BF/typeI_verbatim_cluster_partial.csv
else
  echo 'Extracting ImageNet to local SSD...'
  time tar xf /scratch/public/ImageNet/imagenet.tar -C $TMPDIR
  sync
  echo 'Done.'
  if [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
    python -W ignore -u typeI_imagenet_verbatim.py --preproc cluster --n_seeds 30 --out $BF/typeI_verbatim_cluster_full.csv
  else
    python -W ignore -u typeI_imagenet_verbatim.py --preproc paper --n_seeds 30 --out $BF/typeI_verbatim_paper_full.csv
  fi
fi
rc=$?; echo "task $SLURM_ARRAY_TASK_ID finished rc=$rc"; exit $rc
