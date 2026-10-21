# Saved figure inputs

These are numerical inputs and reference assets, not plotting programs.

- `fig_data_s50/`: per-epoch summaries, onset records and synthetic repetitions.
  `fig4a_matched.csv` supplies the matched synthetic comparisons in Figure 3(a).
- `benchmark_final/`: supervised training and test-loss records.
- `synthetic/`: additional synthetic measurements.
- `manuscript/`: supplementary counts and reference figure assets.

The per-seed test records are in `../data/`; `results.py` recomputes their
summaries. SimCLR loss records are in `../data/analyses/contrastive_loss/`. Historical filenames
retain their original figure numbering, which differs from the final paper.
