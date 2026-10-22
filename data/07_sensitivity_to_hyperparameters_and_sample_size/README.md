# Appendix E; Table 7

## Hyperparameter sweeps: k and M

The saved logs vary k at M=2,000, then vary M at k=60. Each setting reports rejections out of 50 repetitions.

| Dimension | Sweep log |
| --- | --- |
| 32 | [km_D32.log](records/km_D32.log) |
| 64 | [km_D64.log](records/km_D64.log) |
| 128 | [km_D128.log](records/km_D128.log) |
| 512 | [km_D512.log](records/km_D512.log) |
| 2,048 | [km_D2048.log](records/km_D2048.log) |

The five files with job IDs in their names are identical copies of these logs. These one-dimensional sweeps are not the full k × M grid plotted in Appendix E. The original full-grid outputs, sweep runner and plotting script are not included, so that figure cannot currently be regenerated from this release. `synthetic.py` runs the separate experiments in Section 4.4 and Appendices D and G.

## Sample size: Table 7

[small_sample.csv](small_sample.csv) summarizes 1,500 points per split, including additional train-mode rows. Per-seed tests are shared with [Sections 4.2–4.3](../01_main_results/records/), [Section 4.4](../03_synthetic_experiments_and_feature_extraction/records/) and [Appendix B](../04_nearest_neighbor_two_sample_baselines/records/).

For new tests on extracted embeddings, follow the [checkpoint commands](../../REPRODUCIBILITY.md#new-tests-on-feature-arrays) with `--budget 1500`.

[Filename and column guide](../README.md#names-used-in-the-files) explains `bntrain`, `raw`, budgets, checkpoints and seeds.
