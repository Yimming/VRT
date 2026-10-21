# Environment

All results were produced on the Hábrók cluster of the University of Groningen inside one Python virtual environment
on top of the module `PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1` (full module list in `env/module_list.txt`,
full `pip freeze` in `env/pip_freeze_venv311.txt`; that freeze is an archival record with cluster-local paths, not a
portable install command; `requirements.txt` carries the portable pins).

| component | version |
|---|---|
| Python | 3.11.3 (GCCcore 12.3.0) |
| NumPy | 1.26.4 |
| SciPy | 1.17.0 |
| scikit-learn | 1.8.0 |
| PyTorch | 2.1.2 (CUDA 12.1.1, cuDNN 8.9.2.26), from the module |
| torchvision | 0.16.2, from the module |
| matplotlib | 3.10.0 |
| threadpoolctl | 3.6.0 |
| mmdagg | git commit baca0a989273b0961b81135b4488fb5695a9e7ab of github.com/antoninschrab/mmdagg (the `mmdagg.np` backend) |
| BLAS | OpenBLAS 0.3.23 through FlexiBLAS 3.3.1 |
| pypdf | needed only by `figures/manuscript/` (see its requirements.txt) |

Hardware. CPU tasks (`jobs/job_s50_cpu.sh`) ran on standard nodes with two AMD EPYC 7763 processors (64 cores each,
2.45 GHz, no hyper-threading), 8 cores and 32 GB per task with `OMP_NUM_THREADS=8`. GPU tasks requested one GPU of
any type with 4 cores and ran on V100 or A100 nodes as the job logs record (V100 nodes carry two Intel Xeon Gold
6150 processors); `jobs/job_s50_gpu.sh` sets `OMP_NUM_THREADS=4` for the CPU-side tests inside those jobs, the
earlier `job_knn_v2.sh` requested 4 cores without setting it. A resource request or an environment variable does
not by itself verify every BLAS thread pool; the runtime
measurements of Appendix K were taken separately on one standard CPU node with 8 threads, `threadpoolctl` limits
and `CUDA_VISIBLE_DEVICES=""` (`tools/tests_timing.py`, results in `data/timing`). Node types follow the cluster
documentation at wiki.hpc.rug.nl.

Which test ran where. VRT (forward and reversed), C2ST and MMDAgg used NumPy or scikit-learn on the CPU. The GPU tasks
were feature extraction, the on-the-fly extraction of the SimCLR models, the torch backend of MMD and ED, and the
neighbour graph of the nearest-neighbour tests (`tasks/s50_gpu_tasks_*.txt`).

Randomness. The scripts preserve the random-number generators used in the experiments. VRT and the MMD, ED, C2ST
and MMDAgg wrappers use NumPy `RandomState(seed)` for their sampling (subsample draws and permutations); the
reversed VRT draws its budget subsample with `RandomState(1000 + seed)` (`vrt_reversed_budget.py` and the `vrt_rev`
branch of `ssl_s50_fill.py`). The kNN / Henze implementation uses `default_rng(seed)`. Exact reproduction of a
(checkpoint, seed) result also depends on the feature file, the dtype (the main-budget SimCLR VRT of `ssl_s50_fill.py`
runs on float32 features, its small budget and the CIFAR VRT path `vrt_forward_budget.py` on float64) and the
computational backend: the neighbour graph of the kNN tests is computed with a blocked Gram matrix and top-k on the
GPU when one is visible and on the CPU otherwise, and the two differ in float32 near-ties, so kNN p-values are
reproducible per device, not bit-identical across devices (differences of at most 0.016 in p were observed in the
checked checkpoint and seed pairs).

A separate environment with JAX (`numpy 2.4.6, jax 0.10.2`) was used only for the JAX backend timing of MMDAgg in
`tools/mmdagg_timing.py`; it is not needed for any number in the paper. The release tests were run under Python 3.11
to 3.14 with NumPy 1.26 and 2.x; the full model pipeline was not re-executed in a fresh environment for this release.
