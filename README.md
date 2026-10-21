# VRT: Volume Ratio Test

Code and results for **Early Detection of Backbone Divergence via Neighborhood Structure in Learned Embeddings** (NeurIPS 2026).

Yiming Zhang · Reza Zare Hassanpour · Georgi Gaydadjiev

## Run

Use Python 3.11. To reproduce the saved result tables on the CPU:

```bash
git clone https://github.com/Yimming/VRT.git
cd VRT
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install numpy==1.26.4 scipy==1.17.0
python results.py summaries --data data --out results/paper_tables
```

The summary CSVs are written to `results/paper_tables/`; no training or GPU is needed.

To run VRT on your own embeddings, use the [Python example](REPRODUCIBILITY.md#direct-python-use) or [checkpoint commands](REPRODUCIBILITY.md#new-tests-on-feature-arrays).

## Data

[data/](data/README.md) is ordered by paper section (`01`–`11`). For the main results:

- [Summary table](data/01_main_results/main_results.csv): sustained detection epochs.
- [Per-seed test results](data/01_main_results/records/): p-values and MMDAgg rejection decisions, grouped by method.

The [data guide](data/README.md) maps every folder to the paper and explains columns and names such as `bntrain`.

Image datasets, checkpoints and extracted embeddings are not included. See the [reproduction scope](REPRODUCIBILITY.md#scope) for available inputs.

## Code and paper sections

| File | Purpose | Paper |
| --- | --- | --- |
| `vrt.py` | VRT scores and permutation test | Section 3; Appendices A, G |
| `baselines.py` | MMD, ED, C2ST, MMDAgg, Schilling and Henze | Section 4; Appendix B |
| `models.py` | Models and feature extraction | Section 4.1 |
| `train.py` | Training and loss trajectories | Sections 4.1, 4.3; Appendix F |
| `evaluate.py` | Checkpoint tests, geometry and timing | Tables 1–3; Appendices B, E, G–K |
| `controls.py` | Random-network and data-split controls | Appendix C |
| `synthetic.py` | Synthetic experiments | Section 4.4; Appendices D, G |
| `results.py` | Validate data and regenerate tables | Tables 1–9 |
| `common.py` | Shared utilities | Supporting code |

[Full commands](REPRODUCIBILITY.md) · [Plotting scripts](figures/README.md) · [Original scripts and jobs](archive/README.md) · [Citation](CITATION.cff)

Code and data licenses are pending author selection; third-party licenses are preserved in `third_party/`.
