# Environment

Use Python 3.11 for the pinned dependencies in `requirements.txt`. VRT itself
requires only NumPy and SciPy; training and feature extraction also require
PyTorch and torchvision. MMDAgg uses the authors' NumPy implementation at the
commit pinned in `requirements.txt`.

The experiment environment records Python 3.11.3, NumPy 1.26.4, SciPy 1.17.0,
scikit-learn 1.8.0 and PyTorch 2.1.2 with torchvision 0.16.2. PyTorch was supplied
by a CUDA 12.1.1 cluster module. The dependency file contains portable package
pins rather than paths into that installation.

VRT, C2ST and MMDAgg run on the CPU. MMD, ED and the nearest-neighbour graph
support the torch backend, which uses CUDA when available. Feature extraction
and training use CUDA when available. The paper's GPU jobs ran on V100 or A100
nodes; Appendix K reports VRT alone on an AMD EPYC 7763 CPU node with eight
threads, controlled through `threadpoolctl`.

VRT preserves each protocol's sampling and permutation stream. Its main
SimCLR experiment uses float32 features; supervised VRT, reversed VRT and the
small-budget VRT use float64. Most baseline inputs are float32. The kNN tests
use `default_rng`; VRT and the other sampling wrappers use `RandomState`.
Near-ties in float32 nearest-neighbour distances can depend on the device.
Matching a published per-seed result requires the original features, dtype,
seed and computational backend. Training seeds do not guarantee identical
checkpoints across software or hardware versions.
