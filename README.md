# VRT: Volume Ratio Test

Code and per-seed results for **Early Detection of Backbone Divergence via Neighborhood Structure in Learned Embeddings** (NeurIPS 2026).

VRT compares training and test embeddings through their local neighbourhoods. The test runs on the CPU using NumPy and SciPy.

## Setup

Use Python 3.11 and run commands from the repository root. For VRT and the table summaries:

```bash
git clone https://github.com/Yimming/VRT.git
cd VRT
python -m venv .venv
source .venv/bin/activate
python -m pip install numpy==1.26.4 scipy==1.17.0
```

Training, baselines and plotting need additional packages:

```bash
python -m pip install -r requirements.txt
```

The experiment environment is recorded in [ENVIRONMENT.md](ENVIRONMENT.md).

## Run VRT on your embeddings

Save two arrays as `train.npy` and `test.npy`, each with shape `(10000, D)`. Rows must be finite, nonzero feature vectors from the same embedding space.

```python
import sys
import numpy as np

sys.path.insert(0, "code")
from release_utils import validate_features
from vrt_low_lib import l2n, vrt_full

train = np.load("train.npy").astype(np.float64)
test = np.load("test.npy").astype(np.float64)
validate_features(train, test)
assert train.shape == test.shape and train.shape[0] == 10000

statistic, p_value = vrt_full(
    l2n(train), l2n(test), D=train.shape[1], seed=42
)
print(f"statistic={statistic:.6g}, p={p_value:.6g}")
```

This uses 2,000 training reference points, 8,000 disjoint training queries, all 10,000 test queries, 60 neighbours and 2,000 permutations. The return values are the test-minus-training median score difference and its two-sided permutation p-value. This is one test; the paper summarizes 50 analysis seeds at each checkpoint.

For the small-budget protocol, use `vrt_low` with 1,500 points per split. Its defaults are 500 reference points, 1,000 training queries, 1,500 test queries and 15 neighbours, again with 2,000 permutations.

For a sequence of checkpoints, store the two arrays in each `.npz` file as `h_tr` and `h_te`, then run:

```bash
python code/vrt_forward_budget.py \
  --tag my_model --budget large \
  --npz_fmt 'features/ep{ep:03d}.npz' \
  --epochs 0-200 --seeds 42-91 \
  --out results/vrt_fwd_my_model_large.csv
```

Use a separate output file for each model and protocol. The large-budget runner uses every test row, so supply exactly 10,000 test embeddings to match the paper. Missing checkpoint files are skipped.

## Reproduce the paper's tables

The included `data/` contains cleaned per-seed and per-repetition results. No checkpoints or GPU are needed to aggregate them:

```bash
python tools/make_onsets.py --data data --out onsets.csv
```

This validates the result files, prints Tables 1–3 and writes 784 onset records, including the small-budget, nearest-neighbour and reversed-reference comparisons. The reference output is [docs/onsets_final.csv](docs/onsets_final.csv).

For VRT, a checkpoint rejects when the median p-value across seeds 42–91 is below 0.05. Sustained onset is the first evaluated checkpoint at which it rejects and continues to reject at every later evaluated checkpoint. Baseline decision rules and the script-to-table mapping are in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Files

| Path | Contents |
| --- | --- |
| `code/` | VRT, baselines, training and feature-extraction scripts |
| `data/` | Result files, with file hashes and export records in `MANIFEST.csv` |
| `tools/` | Validation, result export and table aggregation |
| `figures/` | Available figure scripts, synthetic results and training logs |
| `jobs/`, `tasks/` | Original cluster commands; adjust paths before reuse |
| `tests/` | Tests for the implementations and result-processing tools |
| `third_party/` | SupContrast training code and its license |

After installing the full dependencies, run `python -m pytest tests -q` to check the release.

The original checkpoints and extracted feature files are not included. Some supplementary figures have saved summaries or figure assets rather than complete generation scripts. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md#status) and the [figure guide](figures/README.md) for the exact scope.

## Citation and license

Paper metadata is in [CITATION.cff](CITATION.cff). Code and data licenses are pending author selection; third-party code retains its existing license.
