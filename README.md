# VRT: Volume Ratio Test

Code and per-seed results for **Early Detection of Backbone Divergence via Neighborhood Structure in Learned Embeddings** (NeurIPS 2026).

VRT tests whether training and test embeddings differ in their local neighbourhood geometry. It runs on the CPU using NumPy and SciPy.

## Setup

Use Python 3.11 and run commands from the repository root:

```bash
git clone https://github.com/Yimming/VRT.git
cd VRT
python -m venv .venv
source .venv/bin/activate
python -m pip install numpy==1.26.4 scipy==1.17.0
```

This is enough for the VRT Python API and the saved-result summaries. For baselines, feature extraction, training and tests:

```bash
python -m pip install -r requirements.txt
```

The original experiment environment is recorded in [ENVIRONMENT.md](ENVIRONMENT.md).

## Reproduce the reported results

Start with [data/summaries/](data/summaries/) for the paper results in CSV form. The [data guide](data/README.md) explains the folders, model names and fields. To regenerate the summaries without retraining models or downloading checkpoints:

```bash
python results.py validate --data data
python results.py summaries --data data --out results/summaries
```

The summaries include all 784 onset records and the supplementary comparisons. `python results.py onsets --data data --out results/onsets.csv` prints Tables 1–3 and reproduces [docs/onsets_final.csv](docs/onsets_final.csv). Further commands are in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Use VRT on your embeddings

For the main protocol, supply at least 10,000 training rows and exactly 10,000 test rows, with the same feature dimension. Each row must be finite and nonzero.

```python
import numpy as np
from common import l2n, validate_features
from vrt import vrt_full

train = np.load("train.npy").astype(np.float64)
test = np.load("test.npy").astype(np.float64)
validate_features(train, test)

statistic, p_value = vrt_full(
    l2n(train), l2n(test), D=train.shape[1], seed=42
)
print(statistic, p_value)
```

This uses 2,000 training reference points, 8,000 disjoint training queries, 10,000 test queries, 60 neighbours and 2,000 permutations. The statistic is the test-minus-training median score difference; the p-value is two-sided. For 1,500 points per split, use `vrt_low` with its default 500 reference points and 15 neighbours.

To evaluate saved checkpoints, store raw feature arrays as `h_tr` and `h_te` in each NPZ file:

```bash
python evaluate.py test \
  --features 'features/ep{ep:03d}.npz' --model my_model \
  --epochs 0-200 --methods vrt --seeds 42-91 \
  --out results/my_model.csv
python results.py new-runs \
  --input results/my_model.csv --out results/my_model_onsets.csv
```

Feature normalization is applied by the runner. The 50 seeds repeat the statistical analysis of each fixed checkpoint; they are not 50 independently trained networks.

## Code map

There are nine main Python modules:

| Module | Purpose | Paper |
| --- | --- | --- |
| `vrt.py` | Neighbourhood scores and forward/reversed VRT | Section 3; Appendices A, G |
| `baselines.py` | MMD, ED, C2ST, MMDAgg, Schilling and Henze | Section 4; Appendix B |
| `evaluate.py` | Checkpoint tests, score statistics, class geometry and timing | Tables 1–3; Appendices B, E, G–K |
| `models.py` | Backbone definitions and feature extraction | Section 4.1 |
| `train.py` | Training recipes and fixed-view SimCLR loss evaluation | Sections 4.1, 4.3; Appendix F |
| `controls.py` | Random-network and data-split controls | Appendix C |
| `synthetic.py` | Cluster, direction, length, bandwidth and reference experiments | Section 4.4; Appendices D, G |
| `results.py` | Validation, export and paper summaries | Tables 1–9 and supplementary results |
| `common.py` | Shared input, numerical and file utilities | Supporting code |

[data/](data/README.md) contains summaries and source records grouped into benchmarks, calibration, synthetic experiments, additional analyses and runtime. `figures/` contains additional saved numerical inputs and figure assets. `tests/` contains regression tests. SupContrast-derived code retains its attribution in `third_party/`.

Original checkpoints, extracted feature arrays and third-party image datasets are not bundled. Saved results reproduce the reported summaries; new training runs need not yield identical checkpoints or detection epochs. The remaining figure-source gaps are listed in [REPRODUCIBILITY.md](REPRODUCIBILITY.md#scope).

## Tests and citation

```bash
python -m pytest tests -q
python evaluate.py smoke
```

Citation metadata is in [CITATION.cff](CITATION.cff). Code and data licenses are pending author selection; third-party code retains its existing license.
