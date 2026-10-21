# Result data

Start with [summaries/](summaries/) for readable CSVs corresponding to the paper tables. The other folders hold individual test results, calibration runs and supporting measurements. Original image datasets, checkpoints and embedding arrays are not included.

## Paper summaries

| File | Contents |
| --- | --- |
| [main_results.csv](summaries/main_results.csv) | Table 1: normalized embeddings, eval extraction |
| [batchnorm_comparison.csv](summaries/batchnorm_comparison.csv) | Table 2: eval versus train extraction |
| [raw_features.csv](summaries/raw_features.csv) | Table 3: unnormalized-feature comparisons |
| [nearest_neighbors.csv](summaries/nearest_neighbors.csv) | Table 4: Schilling and Henze |
| [calibration.csv](summaries/calibration.csv) | Table 5 and Appendix C: single-test rejection counts |
| [synthetic_power.csv](summaries/synthetic_power.csv) | Table 6, Figure 3(a) and Appendix D: synthetic rejection counts |
| [small_sample.csv](summaries/small_sample.csv) | Table 7 and additional train-mode results: 1,500 points per split |
| [reversed_onsets.csv](summaries/reversed_onsets.csv) | Table 8: training versus test reference sets |
| [score_transform.csv](summaries/score_transform.csv) | Appendix I: MMD on VRT scores |
| [runtime.csv](summaries/runtime.csv) | Table 9: VRT CPU runtime |
| [all_onsets.csv](summaries/all_onsets.csv) | All 784 onset cells, including coverage and status |

To regenerate these summaries, run from the repository root:

```bash
python results.py summaries --data data --out results/summaries
```

For Table 6 and Figure 3(a), run `python results.py synthetic --data data`. This also reads [fig4a_matched.csv](../figures/fig_data_s50/fig4a_matched.csv), which is stored with the additional figure inputs.

In the comparison summaries, `null` means no sustained detection; `not evaluated` means no result is available. Ranges span the five zoo models. `<=100` marks detection at the first available Phase Zoo checkpoint. The complete numerical records and coverage fields are in `all_onsets.csv`.

## Source records

| Folder | Contents |
| --- | --- |
| [benchmarks/](benchmarks/) | VRT and baseline tests, grouped by method; SimCLR results are under `simclr/` |
| [calibration/](calibration/) | CIFAR and ImageNet random-network and data-split controls |
| [synthetic/](synthetic/) | Cluster, direction, length and bandwidth experiments; saved hyperparameter logs |
| [analyses/](analyses/) | Reversed-reference tests, score transformations and SimCLR loss curves |
| [runtime/](runtime/) | Timing records; the summary selects the final float64 VRT CPU measurements |

The 5,156 source files keep their original contents and filenames. Checkpoint chunks are combined by `results.py`; they are not separate model training runs. [MANIFEST.csv](MANIFEST.csv) records file hashes and the original export location in `source_path`. It covers source records; `summaries/` contains derived tables. Run `python results.py validate --data data` to validate the records and their hashes.

Some earlier runs remain alongside the final measurements, including `benchmarks/simclr/aggregation_and_earlier_runs/`. The summary commands select the paper protocols.

## Model names

The source files retain these model identifiers:

| Identifier | Model and data |
| --- | --- |
| `v7` | ResNet20-FRN-SiLU, CIFAR-10, D=64 |
| `r50` | ResNet-50, CIFAR-100, D=2,048 |
| `ablA`, `ablB`, `ablC`, `ablD` | ResNet-18, CIFAR-10, D=512: baseline, stronger weight decay, stronger augmentation, both |
| `zoo_c10_m0`–`m4`, `zoo_c100_m0`–`m4` | Five independent ResNet-18 zoo models per dataset, D=512 |
| `phase_bs8`, `phase_bs128`, `phase_bs1024` | ResNet-50 Phase Zoo, CIFAR-100, D=2,048; suffix is the training batch size |
| `simclr_c10`, `simclr_c100` | SimCLR ResNet-50, D=2,048; also recorded as `cifar10` and `cifar100` in SimCLR CSVs |
| `wp` | 5-layer CNN, CIFAR-10, D=32 |
| `imgnet_simple`, `imgnet_adv` | DeepResNet-50, ImageNet, D=2,048; simple or advanced augmentation |

The `_bntrain` suffix means features were extracted using BatchNorm statistics from the current batch. It does not indicate a separately trained network. Calibration files also use architecture names such as `r18_c10`, `r50_c100` and `frn_c10` for randomly initialized models.

## Reading the fields

- `epoch` identifies the checkpoint. `seed` repeats the statistical test on that checkpoint; the main analysis uses seeds 42–91.
- `large` and `small` denote 10,000 and 1,500 accessed points per split. MMDAgg uses 5,000 points per split in the main comparison. `n_perm` records the permutation count where present.
- `p` is a p-value except for MMDAgg, where it records a binary rejection decision. `acc` is C2ST's held-out accuracy. Empty or `nan` accuracy fields for other methods are unused.
- `dc` and `delta_c` are test-minus-training median VRT score differences. `p_exact` and `p_henze_exact`, where present, recover the recorded permutation-p grid.
- A blank `onset` with null status means no sustained detection through the final evaluated checkpoint. Coverage fields distinguish this from incomplete evaluation.
- In `synthetic_power.csv`, `conc_0.3` means an increase of `0.3 * sqrt(D)` in kappa, not 30%. `scale` is the Gaussian bandwidth multiplier; zero for VRT is unused. `NORM-MED` and `NORM-DISP` test vector lengths, while VRT and MMD test normalized directions.

The aggregation rules and commands for new experiments are in [REPRODUCIBILITY.md](../REPRODUCIBILITY.md).
