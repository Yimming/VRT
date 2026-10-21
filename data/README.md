# Data by paper section

Open the section you are reading. Summary CSVs are at the top of each folder; `records/` contains the individual test results. Shared source records are stored once and referenced from the other sections.

| Folder / paper element | Runner | Main fields | Source files |
| --- | --- | --- | --- |
| [4_2_main_results/](4_2_main_results/) — 4.2 Main results; Tables 1–2 | `evaluate.py test` | `epoch, seed, budget, p, acc` | 3,576 |
| [4_3_regularization_interventions/](4_3_regularization_interventions/) — 4.3 Regularization interventions | `train.py regularization; evaluate.py test` | `VRT, MMD, MMDAgg, C2ST, ED` | 0 |
| [4_4_synthetic_experiments_and_feature_extraction/](4_4_synthetic_experiments_and_feature_extraction/) — 4.4 Synthetic experiments and feature extraction; Table 3 | `evaluate.py test --raw` | `epoch, seed, n_samples, p` | 336 |
| [B_nearest_neighbor_two_sample_baselines/](B_nearest_neighbor_two_sample_baselines/) — Appendix B; Table 4 | `evaluate.py test` | `epoch, seed, k, p_schilling, p_henze` | 168 |
| [C_type_i_calibration_and_controls_before_training/](C_type_i_calibration_and_controls_before_training/) — Appendix C; Table 5 | `controls.py` | `arch, split, bn_mode, rep, p, reject` | 669 |
| [D_synthetic_power_profile_details/](D_synthetic_power_profile_details/) — Appendix D; Table 6 and Figure 3(a) | `synthetic.py` | `cond, C, budget, rep, test, p, scale` | 4 |
| [E_sensitivity_to_hyperparameters_and_sample_size/](E_sensitivity_to_hyperparameters_and_sample_size/) — Appendix E; Table 7 | `evaluate.py test; synthetic.py` | `mode, budget, onset; sweep logs` | 10 |
| [F_per_model_trajectories/](F_per_model_trajectories/) — Appendix F; Figures 2 and 6 | `train.py evaluate-loss` | `epoch, split, loss` | 20 |
| [G_reference_set_selection/](G_reference_set_selection/) — Appendix G; Table 8; also Appendix H | `evaluate.py test --methods reversed` | `epoch, seed, direction, M, k, delta_c, p` | 92 |
| [I_sensitivity_advantage_of_score_transformation/](I_sensitivity_advantage_of_score_transformation/) — Appendix I | `evaluate.py test; evaluate.py score-statistics` | `epoch, seed, p, score statistics` | 272 |
| [K_compute_resources_and_runtime/](K_compute_resources_and_runtime/) — Appendix K; Table 9 | `evaluate.py timing` | `model, budget, seed, seconds` | 9 |

Counts cover the 5,156 source files in [MANIFEST.csv](MANIFEST.csv), excluding generated summaries and README files. Section 4.3 uses the shared Section 4.2 records. The complete [784-cell onset table](4_2_main_results/all_onsets.csv) includes all models, methods, extraction modes and budgets.

## Names used in the files

| Name | Meaning |
| --- | --- |
| `bntrain`, or extraction `mode=train` | BatchNorm uses statistics from the current extraction batch. Checkpoint weights stay fixed: this does not retrain the network. |
| `eval`, or no `_bntrain` suffix in the main benchmark filenames | BatchNorm uses the running statistics saved during training. |
| `raw` | Unnormalized embedding vectors. `l2` means each vector was normalized to unit length. Normalization and BatchNorm extraction mode are separate settings. |
| `large`, `small` | 10,000 or 1,500 accessed points per split. Main MMDAgg uses 5,000 points per split. |
| `n10000`, `b10000`, `b5000` | Number of accessed points per split, not feature dimension or training batch size. |
| `c0-83`, `e525`, `ep061` | Checkpoint/epoch range or a single checkpoint. Use the CSV `epoch` column for the actual evaluated checkpoints. |
| `s42-91`, `seed` | Statistical-test repetitions on fixed checkpoints. The main analysis uses 50 seeds, 42–91; these are not 50 independently trained networks. |
| `p2000`, `n_perm=2000` | 2,000 label permutations where the method uses permutation calibration. |
| `M`, `k` | Number of reference points and nearest neighbours in VRT; for the graph baselines, `k` is the graph neighbour count. |
| `fwd`, `rev`, `test_ref` | Standard training-reference or reversed test-reference VRT. |
| `trtr`, `disjoint_training` | Comparison of two disjoint subsets of training data. |
| `resplit`, `official` | Random partitions or the original dataset training/test split. |
| `null`, `rand`, `init0` | Control conditions; see the relevant CSV and Appendix C for the random-network or data-split setting. |
| `part`, `fast`, `sslagg` | Historical job-chunk/runner names. They are not additional model architectures. `sslagg` is a SimCLR MMDAgg run. |
| `ssl`, `chat` | Self-supervised learning; the ASCII filename spelling of the VRT score c-hat. |

Method names: `vrt` is the Volume Ratio Test; `mmd` is maximum mean discrepancy; `ed` is Energy Distance; `c2st` is the classifier two-sample test; `mmdagg` is aggregated MMD. `knn` stores the nearest-neighbour graph tests of Schilling and Henze.

For example, `vrt_fwd_r50_bntrain_large_c0-83.csv` contains standard-reference VRT on ResNet-50/CIFAR-100, using current-batch BatchNorm statistics, 10,000 points per split, and checkpoints in epochs 0–83. The file is one batch of statistical tests on saved checkpoints.

## Model identifiers

| Identifier | Model and dataset |
| --- | --- |
| `v7` | ResNet20-FRN-SiLU, CIFAR-10, D=64 |
| `r50` | ResNet-50, CIFAR-100, D=2,048 |
| `ablA` / `ablB` / `ablC` / `ablD` | ResNet-18, CIFAR-10, D=512: baseline / 10× weight decay / strong augmentation / both |
| `zoo_c10_m0`–`m4`, `zoo_c100_m0`–`m4` | Five ResNet-18 zoo models per dataset, D=512 |
| `phase_bs8`, `phase_bs128`, `phase_bs1024` | ResNet-50 Phase Zoo, CIFAR-100, D=2,048; `bs` is the training batch size |
| `simclr_c10`, `simclr_c100`; also `cifar10`, `cifar100` in SimCLR files | SimCLR ResNet-50, D=2,048 |
| `wp` | 5-layer CNN, CIFAR-10, D=32 |
| `imgnet_simple`, `imgnet_adv` | DeepResNet-50, ImageNet, D=2,048; simple or advanced augmentation |
| `r18_c10`, `r50_c100`, `frn_c10` in calibration files | Architecture/dataset labels for the randomly initialized controls |

## Reading results

- `p` is a p-value except for MMDAgg, where it is a binary rejection decision. `acc` is C2ST held-out accuracy; unused accuracy fields for other methods may be empty or `nan`.
- `dc` and `delta_c` are test-minus-training median VRT score differences. `p_exact` and `p_henze_exact` recover the recorded permutation-p grid.
- In summary tables, `null` means no sustained detection; `not evaluated` means no available result. Zoo ranges span five models. `<=100` means detection at the first Phase Zoo checkpoint.
- A blank numerical `onset` with null status is distinct from incomplete coverage; the all-onsets file includes both coverage and status.
- In synthetic results, `conc_0.3` means an increase of `0.3 * sqrt(D)` in kappa, not 30%. `scale` is a Gaussian bandwidth multiplier; zero for VRT is unused. `NORM-MED`/`NORM-DISP` test vector lengths, while VRT/MMD test normalized directions.

## Rebuild and provenance

```bash
python results.py validate --data data
python results.py summaries --data data --out results/paper_tables
```

Original source bytes and filenames are preserved. The manifest differs from the original release only in its `path` column; [data_layout.csv](../docs/data_layout.csv) maps historical folders to the paper sections. [EXCLUDED.txt](EXCLUDED.txt) records the two excluded obsolete result directories. Earlier ten-seed SimCLR files are retained under `4_2_main_results/records/simclr/aggregation_and_earlier_runs/` and excluded from final onset summaries.

The [archived scripts and task commands](../archive/README.md) document the original execution workflow. Dataset images, checkpoints and embedding arrays are not bundled. See [REPRODUCIBILITY.md](../REPRODUCIBILITY.md#scope) for missing original figure inputs and additional controls.
