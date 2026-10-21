# Early Detection of Backbone Divergence via Neighborhood Structure in Learned Embeddings

This repository contains the evaluation code and the available reproduction materials for the camera-ready version
of the paper (NeurIPS 2026, submission 27102): the implementations of the Volume Ratio Test (VRT) and of the
baselines (MMD, MMDAgg, C2ST, Energy Distance, and the Schilling and Henze nearest-neighbour tests), the job and task
files with which they were run on the cluster, tools that recompute the tables from the per-seed result files, the
figure scripts with their data, and the per-seed result files (`data/`; see "Status" at the end for what is still
missing).

## Layout

```
code/        runners, test implementations, feature extraction, training, harvest and diagnostic scripts (flat)
jobs/        SLURM job scripts (array jobs; the task line of a job is the line (array index + 1) of a task file)
tasks/       task files: one command line per array task, exactly as run
tools/       release tools: export_results.py (validated result export), result_schemas.py (what a valid result file
             is), make_onsets.py (tables from the data), tests_timing.py and mmdagg_timing.py (Appendix K),
             clean_chatstats_dups.py, fill_train1500_status.py, check_null_csv.py, timing_summary.py
tests/       pytest suite for the release fixes, the guards and the tools (python -m pytest tests -q, 30 tests)
figures/     figure scripts and the data they read; figures/manuscript/ holds the final builds of the four
             appendix figures (see figures/README.md)
data/        result files, 5,156 files / 276 MB, produced by tools/export_results.py (see "Result data" below;
             docs/onsets_final.csv is tools/make_onsets.py on this folder)
env/         pip freeze, module list and folder listings of the cluster environment
third_party/ SupContrast (the SimCLR training code the SSL models were trained with)
docs/        audit material: table_comparison_1017.csv (every value of Tables 1 to 3 against the data),
             patches_vs_asrun_1018.diff (every change relative to the as-run scripts), RELEASE_CHECKS_1018.md
```

## Protocol of the main tables

Tables 1 to 3 use 50 analysis seeds, 42 to 91. Each main-budget test uses 10,000 embeddings per split, except
MMDAgg, which uses 5,000. VRT, MMD, Energy Distance and Henze use 2,000 permutations. C2ST and Schilling use their
stated normal approximations. MMDAgg uses B1 = B2 = 2,000 and B3 = 50. The small-budget comparison (Appendix E)
uses 1,500 embeddings per split. Features are l2-normalized except in Table 3 (unnormalized). A checkpoint rejects
when the median p-value over the seeds is below 0.05 (C2ST additionally needs median held-out accuracy above 0.55;
MMDAgg, which returns a binary decision per seed, rejects when at least 60 percent of the seeds reject). The
detection onset of a cell is the first evaluated checkpoint from which every later evaluated checkpoint rejects (the
sustained onset). For every cell of Tables 1 to 3 the checkpoints between the last non-rejecting checkpoint and the
onset and every saved checkpoint after the onset were evaluated, so those onsets are exact on the saved-checkpoint
grid. The calibration, synthetic and diagnostic experiments have their own repetition counts, stated with each
item below. `tools/make_onsets.py` applies the decision rule above to the result files and prints Tables 1 to 3.

## Reproducing the tables from the result data

```
pip install -r requirements.txt            # or use the versions in ENVIRONMENT.md
python -m pytest tests -q                   # 30 tests
python tools/make_onsets.py --data data --out onsets.csv
```

`make_onsets.py` first validates every result file it reads (`tools/result_schemas.py`: file names, row membership,
integer epoch and seed keys, admissible p-values and decisions, permutation counts that reproduce p, the fixed protocol
values of the runs such as direction, M, k and n_perm, and no key with two different values across files) and stops on
any violation, so that an invalid value can never become a silent "not rejected" and rows of another protocol are never
merged into a cell. A cell whose epochs all lack a seed is listed with the status "no complete epoch" rather than
dropped. It then writes one row per cell (model, extraction
mode, statistic, budget, number of evaluated epochs, onset, unevaluated checkpoints before and after the onset,
epochs with a median p of exactly 0.05, epochs with fewer than 50 seeds) and prints Tables 1 to 3 in the layout of
the paper (ranges over the five zoo models, number of detecting models in brackets when fewer than five detect,
`null` for no sustained detection; the paper prints the Phase Zoo value 100 as "<= 100", its earliest checkpoint).
Table 1 marks the two ImageNet rows with "BN" (MMD and ED onset 1 under eval-mode extraction, no detection under
train-mode extraction, Section 4.4) and "*" (forward VRT null, reversed VRT rejects, Appendix G); the tool prints
the plain onsets. `docs/table_comparison_1017.csv` is the independent check of October 17: all 280 values of Tables
1 to 3 against the same data.

Where every number of the paper comes from (current numbering of the camera-ready):

| paper element | script | data |
|---|---|---|
| Tables 1, 2 (onsets, eval and train mode) | `tools/make_onsets.py`; as-run harvesters `code/harvest_s50.py`, `code/harvest_bn_table.py` | `vrt_fwd`, `mmd_p2000`, `ed_p2000`, `mmdagg`, `c2st`, `bntrain_*`, `ssl`, `ssl_bntrain`, `ssl_baselines` |
| Table 3 (unnormalized features) | `tools/make_onsets.py` rows `mmd_raw`, `ed_raw`; `code/harvest_raw_table.py` | `mmd_raw_p2000`, `ed_raw_p2000`, `ssl/ssl_*_raw_*` |
| Table 4, Appendix B (Schilling, Henze) | `tools/make_onsets.py` rows `schilling`, `henze` (not laid out as a table by the tool) | `knn_v2` |
| Table 5, Appendix C (Type-I checks on CIFAR) | `code/cifar_null_vrt.py`, `cifar_null_base.py`, `cifar_null_rule.py`; harvest `harvest_cifar_null*.py` | `data/null_cifar_vrt`, `data/null_cifar_base`, `data/null_cifar_rule` (per-run files included) |
| ImageNet random-initialization controls (additional release data) | `code/typeI_imagenet_verbatim.py`, `imagenet_randinit_extract.py`, `imagenet_trtr_extract.py`; harvest `harvest_null.py data/null` and `harvest_null.py data/null_trtr _trtr` | `null` (train vs validation), `null_trtr` (disjoint training samples) |
| Table 6, Appendix D (synthetic alternatives) and Figure 3(a) | `code/synth_fig4a_clusters.py`, `synth_fig4a_matched.py`, `mmd_bandwidth_control.py`; numbers `figures/appD_numbers.py` | `figures/fig_data_s50/fig4a_clusters.csv`, `fig4a_matched.csv`, `data/synth/mmd_bandwidth.csv` |
| Table 7, Appendix E (small budget) | `tools/make_onsets.py` rows with budget 1500 | same families, files `*_n1500_*`, `*_small_*`, `*_b1500_*` |
| Table 8, Appendix G (reversed VRT) | `tools/make_onsets.py` rows `vrt_rev` (exact p, see Fixes 3) | `reversed` |
| Table 9, Appendix K (VRT runtime) | `tools/tests_timing.py`, `tools/timing_summary.py` | `timing` |
| Appendix I (MMD on the c-hat scores) | `code/mmd_on_chat_s50.py`, `ssl_mmd_on_chat.py`, `chat_stats_s50.py`; summaries `onc_summary.py`, `chat_stats_summary.py` | `mmd_on_chat`, `chat_stats` |
| Figure 2 (regularization, file `fig3_regularization.pdf`) | `figures/plot_fig3_s50.py` | `figures/fig_data_s50` |
| Figure 3 (synthetic power and ResNet-50 eval vs train, file `fig4_diagnosis.pdf`) | `figures/plot_fig4_s50.py` | `figures/fig_data_s50` |
| Figure 4, Appendix C (`fig_cifar_official_control.pdf`) | `figures/manuscript/appendix_c/` | aggregate counts in the package, verified against the `official` rows in `data/null_cifar_vrt` and `data/null_cifar_base` |
| Figure 6, Appendix F (`fig_trajectories_appendix.pdf`) | `figures/manuscript/appendix_f/` | loss logs, `ssl` and `ssl_nll` |
| Figure 7, Appendix G (`fig_ref_asymmetry.pdf`) | `figures/manuscript/appendix_g/` (vector-asset rebuild, see its README) | |
| Figure 8, Appendix H (`fig_dc_trajectory.pdf`) | `figures/plot_dc_trajectory_s50.py` | `figures/fig_data_s50/dc_*.csv` |
| Figure 9, Appendix I (`fig_mmd_on_chat.pdf`) | `figures/manuscript/appendix_i/` | onset summary in the package |
| Figures 5 and 10 (`fig_km_sweep.pdf`, `fig_decompose.pdf`) | unchanged from the submission; generators not in this tree (see Status) | |
| SimCLR loss curves (Appendix F) | `code/ssl_nll.py` | `ssl_nll` |

## Result data

`data/` is produced from the raw result folders of the cluster with

```
python tools/export_results.py --src <raw s50 folder> --dst data        # dst must be a new or empty folder
```

The export validates every CSV against `tools/result_schemas.py` and stops on the first violation (a file name no
schema recognises, a row that belongs to another model or budget than its file, a non-integer epoch or seed, a
permutation p-value outside (0, 1], an MMDAgg decision other than 0 or 1, a C2ST accuracy outside [0, 1], a count that
does not reproduce p, an exact-p column that does not match p, a key with two different values anywhere in the export).
The export is written to a partial folder beside the destination and moved into place when complete, so a refused
export leaves nothing behind.
It keeps the folder layout and cleans mechanically, logging each change in `data/MANIFEST.csv` (rows, bytes, sha256,
what was removed): NUL bytes (eleven as-run files hold zero-filled regions between rows, left by concurrent appends of
two array tasks to one file; the rows themselves are complete), torn lines (none in the as-run data), rows repeated
verbatim (one file), repeated header lines. It adds `p_exact` to `reversed/` and `p_henze_exact` to `knn_v2/`, whose
runners wrote p with four decimals (see Fixes 3). `data/EXCLUDED.txt` lists the folders left out: `c100_nan_bug`
(SimCLR CIFAR-100 MMD on c-hat before the +inf cap), `chat_stats/dup_backup` (files before the duplicate-row cleanup),
`knn_v2/_superseded` and `null10` (10-seed runs superseded by the 50-seed ones). The 10-seed-stage SimCLR files in
`ssl_baselines/` (`sslbase_*`, `sslvrt_*`, `sslseeds_*`) are copied verbatim and marked as legacy; the harvest never
reads them.

Families and the runners that wrote them (`<tag>` is the model, `_bntrain` marks train-mode BatchNorm extraction,
`c<lo>-<hi>` the epoch chunk of one array task):

| folder | files | runner | columns |
|---|---|---|---|
| `vrt_fwd` | `vrt_fwd_<tag>_{large,small}_c*.csv` | `vrt_forward_budget.py` | tag, budget, epoch, seed, dc, p |
| `reversed` | `vrt_rev_<tag>_{large,small}_c*.csv` | `vrt_reversed_budget.py` | tag, budget, direction, epoch, seed, M, k, delta_c, p (4 decimals), p_exact |
| `mmd_p2000`, `mmd_raw_p2000` | `mmd_<tag>_n{10000,1500}_c*.csv` | `mmd_tst_fast.py` | tag, n_samples, n_perm, epoch, seed, mmd2, p |
| `ed_p2000`, `ed_raw_p2000` | `ed_<tag>_n{10000,1500}_c*.csv` | `energy_tst_fast.py` | tag, n_samples, n_perm, epoch, seed, ed_val, p |
| `knn_v2` | `knn_<tag>_n<N>_part_c*.csv` | `knn_tst_budget.py` | tag, n_samples, k, epoch, seed, p_schilling, p_henze (4 decimals), p_henze_exact |
| `c2st`, `bntrain_c2st` | `c2st_<tag>_b<budgets>_c*.csv` | `base_fill.py --stat c2st` | model, epoch, budget, stat, seed, acc, p |
| `mmdagg`, `bntrain_mmdagg` | `mmdagg_<tag>_b<N>_c*.csv` | `base_fill.py --stat mmdagg` | model, epoch, budget, stat, seed, acc (nan), p (0/1 reject) |
| `ssl`, `ssl_bntrain` | `ssl_<ds>[_bntrain]_<group>_c*.csv` | `ssl_s50_fill.py` (groups vrt, fast, base, slow, raw, knn, rev, extra) | model, epoch, budget, stat, seed, acc, p |
| `ssl_baselines`, `ssl_bntrain` | `sslagg_<ds>[_bntrain]_s42-91_e<epochs>.csv`, `sslagg_<ds>_<lo>-<hi>_e<N>.csv` | `ssl_mmdagg5k_seeds.py` (MMDAgg at 5,000 points) | same, p = 0/1 reject |
| `mmd_on_chat` | `mmdchat_<tag>_c*.csv` | `mmd_on_chat_s50.py`, `ssl_mmd_on_chat.py` | tag, epoch, seed, dc, mmd_c, p[, vrt_p, n_inf] |
| `chat_stats` | `chatstats_<tag>_c*.csv` (+ `.npz` score dumps) | `chat_stats_s50.py` | tag, epoch, seed, stat, obs, p |
| `ssl_nll` | `ssl_nll_<ds>_c*.csv` | `ssl_nll.py` | epoch, tr_eval, te_eval, gap_eval, tr_train, te_train, gap_train |
| `timing` | `tests_timing_<tag>_ep<N>_<run>.csv` | `tools/tests_timing.py` | tag, test, budget, backend, device[, threads], seed, seconds, result |
| `synth` | `fig4a_clusters.csv`, `fig4a_all.csv`, `fig4a_radial.csv`, `mmd_bandwidth.csv` (`fig4a_matched.csv` is in `figures/fig_data_s50`) | `synth_fig4a_*.py`, `synth_radial_fig4a.py`, `mmd_bandwidth_control.py` | cond[, C, budget, scale], test, rep, p[, sigma, d_within, d_between] |
| `null`, `null_trtr` | ImageNet random-initialization controls: train vs validation (`null`) and disjoint training samples (`null_trtr`); `typeI_verbatim_*.csv` (seed, dc, p) and subfolders that mirror the main families (tags `imgnet_rand_{eval,train}[_trtr]`) | `typeI_imagenet_verbatim.py`, `imagenet_randinit_extract.py`, `imagenet_trtr_extract.py` and the main runners | as the main families |
| `null_cifar_vrt` | CIFAR random-initialization checks: `vrt_null_*` (Table 5 random partitions and Figure 4 official split) | `cifar_null_vrt.py` | arch, bn_mode, rep, net_seed, split_seed, vrt_seed, split, budget, D, dc, p, cos, feat_sec |
| `null_cifar_base` | CIFAR baseline checks: `base_null_*` (Table 5 random partitions and Figure 4 official split) | `cifar_null_base.py` | arch, bn_mode, rep, net_seed, split_seed, seed, split, method, budget, p, acc, sec |
| `null_cifar_rule` | CIFAR median-p decision-rule control: `vrt_rule_*` (Appendix C text) | `cifar_null_rule.py` | arch, bn_mode, net, net_seed, budget, seed, dc, p |
| `official_check` | additional pixel-feature and shuffled-extraction controls, 100 repetitions per setting (not the Figure 4 count source) | `cifar_official_check.py` | dataset, features, rep, split, method, p |
| `km_logs` | `km_D*.log` | k x M sweep logs of the submission (Figure 5) | text logs |

Files written by the released runners carry additional columns that the as-run files lack: `features` (`l2` or `raw`)
in the MMD, ED, kNN, C2ST and MMDAgg files, and the integer permutation count with `n_perm` in the reversed VRT and kNN
files; the schemas accept both forms.

## Running the tests on features

The runners read per-epoch feature files `<tag>/ep<NNN>.npz` with the training and test features of one checkpoint
(keys `h_tr`/`h_te`, `X_tr`/`X_te` or `tr`/`te`); `code/extract_feats.py`, `extract_bntrain*.py`,
`imagenet_extract.py` and `bntrain_extract.py` write them from the checkpoints, and the SimCLR runner
`ssl_s50_fill.py` extracts on the fly. The feature files and the checkpoints are too large for this repository and
are not part of it; the training scripts are `train_resnet50_cifar100_seed.py`, `train_regularization_seed.py`
(Configurations A to D) and `train_simclr_supcon.py` (SimCLR on top of `third_party/SupContrast`). Retraining with
these recipes reproduces the protocol, not bit-identical checkpoints or onsets.

The task files in `tasks/` give, for every result file of the 50-seed runs, the exact command with which it was
produced (`s50_make_tasks.py` generated the main lists); `jobs/job_s50_cpu.sh` and `jobs/job_s50_gpu.sh` are the
array jobs that ran them. Three kinds of scripts live side by side in `code/`: the archived historical runners and
task lists (kept as run, including the 10-seed stage, see "Superseded scripts"), the current entry points
(`vrt_forward_budget.py`, `vrt_reversed_budget.py`, `mmd_tst_fast.py`, `energy_tst_fast.py`, `knn_tst_budget.py`,
`base_fill.py`, `ssl_s50_fill.py`, `ssl_mmdagg5k_seeds.py`, `mmd_on_chat_s50.py`, `ssl_mmd_on_chat.py`,
`chat_stats_s50.py`; the published commands pass `--seeds 42-91` explicitly, the defaults `42-51` of some runners are
those of the earlier stage), and, among the entry points, those whose resume logic was hardened for this release
(the first six, see Fixes 2 and 3: a row counts as done only when it belongs to the current run and carries valid
values). Default paths in the scripts and the task files are the cluster locations as run
(`/home3/p318770/venvs/overfitting/benchmark_final/s50` for results, `/scratch/p318770/...` for code, features and
checkpoints); the harvest and check scripts take the result folder from the environment variables `S50` and `SSLB`
(`SSLB` is the folder with the SimCLR eval-mode MMDAgg files, `data/ssl_baselines`).

`code/find_grid_gaps.py` is the completeness check used during the runs (which saved checkpoints between the last
non-rejecting epoch and the onset, or after the onset, were not yet evaluated), with `--tails` for the checkpoints
after the onset; `tools/fill_train1500_status.py` reports the small-budget train-mode tier.

## Fixes relative to the as-run scripts

All numbers in the paper were produced by the scripts as they were run on the cluster (the versions in the task
files; `docs/patches_vs_asrun_1018.diff` lists every change made since). Two audits (October 17 and 18) found the
problems below in those scripts. None of them changed a published value except the rounding case in item 3, which
is corrected in the paper (Appendix G, median lag 0.5 epochs). For valid inputs the statistical tests compute
exactly what they computed (same arithmetic, same random stream; checked on 30 old/new pairs in the second review);
the training script's resume fix (item 5) is a separate change to a script that produced no published number through
its resume path. Everything is covered by `tests/`.

1. Non-finite inputs, scores or statistics. A NaN in the features propagated to a NaN statistic; every comparison
   with the permuted statistics was False and the test returned p = 1/2001 as if it had rejected. In the VRT median
   tests the same happened when the median difference itself was NaN or infinite (all scores of a side +inf, which
   float32 duplicates can produce). `release_utils.validate_features`, `check_scores`, `check_statistics` and
   `check_perm_statistic` now stop the run in every active path: `vrt_low_lib.vrt_low` and `vrt_full`,
   `supcon_fill_vrt.vrt_test` (the SimCLR path of `ssl_s50_fill.py`), `vrt_reversed_budget.perm_p`,
   `mmd_tst_fast.py`, `energy_tst_fast.py`, `baselines2_stats.py` (C2ST, MMDAgg), `vrt_forward_budget.py`,
   `base_fill.py`. Isolated +inf scores are allowed by the median tests as long as the observed and every permuted
   median difference stay finite (an infinite or NaN difference stops the run); a NaN score (a query coinciding with
   duplicate reference points) is refused. The one place where a non-finite statistic occurred in the as-run data
   (SimCLR CIFAR-100 MMD on c-hat, float32 duplicate queries giving +inf scores) was rerun with the +inf cap before the
   paper's numbers were taken; the affected files are excluded from `data/`. The recorded SimCLR MMD-on-c-hat runs show
   at most 2 of 8,000 to 10,000 queries with +inf, far below the fraction that would make a group median infinite; the
   stored forward-VRT statistics (`dc`) are all finite. The permuted statistics of the historical runs were not logged,
   so this is a guard for future runs, not a proof about every past permutation.
2. Resume. Several runners counted the rows of an epoch and reran it when the count was short, which duplicated
   seeds. `ResumeIndex` keys on (epoch, seed) or (epoch, budget, seed) and counts a row as done only when it belongs
   to the current run (tag, statistic, sample budget, permutation count, feature normalization) and carries valid
   values (finite p in [0, 1], a 0/1 MMDAgg decision, an accuracy in [0, 1], an integer permutation count that
   reproduces p); rows of another run, conflicting duplicates and invalid rows make the resume refuse (an output file
   holds one run; write to a new path instead of mixing runs), and a torn last line is moved into a `.torn` sidecar
   before anything is appended. `open_append` refuses an output file whose header differs from the runner's schema.
   The runners refuse a sample budget larger than a split instead of silently testing fewer points, and record the
   feature normalization in a new `features` column (`l2` or `raw`), so a resumed file is identified by its full
   configuration. Applied in `mmd_tst_fast.py`, `energy_tst_fast.py`, `knn_tst_budget.py`, `vrt_reversed_budget.py`,
   `base_fill.py`. The as-run files contain
   repeated rows in one place (`mmdagg/mmdagg_r50_b5000_c149-149.csv`, identical rows, removed by the export) and the
   c-hat diagnostic files were cleaned with `tools/clean_chatstats_dups.py` (each key once, backups in
   `chat_stats/dup_backup`, excluded from `data/`).
3. Four-decimal p-values. `vrt_reversed_budget.py` and `knn_tst_budget.py` wrote p with four decimals; the exact value
   (count + 1)/2001 is recovered as round(p x 2001)/2001 (`release_utils.exact_perm_p`, which refuses values that are
   not within rounding distance of the lattice; columns `p_exact` and `p_henze_exact` in `data/`). The runners now
   write six decimals and the integer count. One decision depended on it: reversed VRT of zoo CIFAR-10 model 3 at
   epoch 30 (stored median 0.0500, exact 0.049975), which moves that onset from 31 to 29 and the median lag of the
   reversed VRT behind the forward VRT over the ten zoo models from 1.0 to 0.5. Two Henze epoch decisions change
   without changing an onset.
4. Unnormalized features. The `--raw` entry points of `mmd_tst_fast.py`, `energy_tst_fast.py`, `knn_tst_budget.py`
   and `base_fill.py` accept zero rows (valid points of the raw feature space; the zero-norm check applies only when
   features are l2-normalized).
5. Training resume index. `train_resnet50_cifar100_seed.py` resumed from `checkpoint_N` with loop index N + 1 and
   wrote `checkpoint_000` as a bare state dict; it now resumes with index N (`release_utils.resume_start_epoch`),
   writes a full dictionary for every checkpoint and saves and restores the RNG state. The training logs reviewed
   for the seed reruns show runs that completed in one job; whether every historical training run did is not
   documented, so this is recorded as a fix of the script, not as a statement about the published checkpoints.

Also corrected: the docstring of `ssl_mmd_on_chat.py` attributed the +inf scores to NumPy 2 casting; the runs used
NumPy 1.26.4, where the 1e-300 floor of `log_cap_volume` is cast to the float32 array dtype and becomes 0 under
value-based casting as well (`tests/test_release_fixes.py` documents the behaviour). The data tools
(`export_results.py`, `make_onsets.py`) validate their inputs as described above instead of skipping unparsable rows.

## Superseded scripts

`ssl_baselines_fill.py`, `ssl_mmdagg5k_fill.py`, `ssl_seeds_fill.py`, `ssl_vrt_low_fill.py` and `supcon_fill_vrt.py`
(as a runner) belong to the 10-seed stage and resume by row count; their header says so. `supcon_fill_vrt.py` stays
in place because `ssl_s50_fill.py`, `ssl_mmd_on_chat.py`, `ssl_nll.py`, `bntrain_extract.py`, `cifar_null_vrt.py` and
others import its extraction and `vrt_test` functions (the latter now guarded). Scripts with a `LEGACY` header
(`harvest_base.py`, `harvest_ed.py`, `harvest_p2000.py`, `harvest_ssl.py`, `harvest_ssl_seeds.py`, `harvest_bntrain.py`,
`compare_ablB.py`, `s50_check.py`, `ksweep15_fill.py`, `unnorm_stats.py`, `supcon_extract_vrt.py`) read the files of
the 10-seed stage or compare the two stages; historical task files and `gpufill*.py` still reference some of them,
so they stay where they are. The published numbers do not depend on them; the current entry points are listed under
"Running the tests on features".

## Environment

See `ENVIRONMENT.md` (Python 3.11.3, NumPy 1.26.4, SciPy 1.17.0, scikit-learn 1.8.0, PyTorch 2.1.2 with CUDA 12.1.1,
mmdagg at commit baca0a9, OpenBLAS 0.3.23 through FlexiBLAS 3.3.1, the random-number generators each test uses, and
the hardware) and `env/pip_freeze_venv311.txt` (archival record of the cluster environment, not a portable install
command; `requirements.txt` holds the portable pins). VRT itself runs on the CPU in NumPy; the GPU was used for
feature extraction, SimCLR on-the-fly extraction, the torch backend of MMD and ED and the neighbour graph of the kNN
tests.

## Status

Included results and remaining items:

- `data/` holds the result set exported on October 19 (5,156 files, 5,061,086 validated records, `data/MANIFEST.csv`);
  `tools/make_onsets.py --data data` gives 784 cells, every detecting cell final (all checkpoints between the last
  non-rejecting one and the onset, and all after the onset, evaluated) and every null cell evaluated at its last
  checkpoint (`docs/onsets_final.csv`, `docs/onsets_final_stdout.txt`); Tables 1 to 3, the kNN table, the small-budget
  table and the reversed-VRT table of the paper agree with it entry by entry.
- Appendix K: the VRT seconds come from `data/timing/tests_timing_{r50_ep61,v7_ep38}_vrt64.csv` (float64
  `vrt_full` at 10,000 points and `vrt_low` at 1,500 points, one standard CPU node, 8 threads, median over ten seeds):
  2.36 and 1.86 s at 10,000 points, 0.28 and 0.25 s at 1,500 points; the manuscript table uses these values.
- The CIFAR per-run results are included in `data/null_cifar_vrt`, `data/null_cifar_base` and `data/null_cifar_rule`.
  The 25 Table 5 values and all 25 Figure 4 aggregate counts match these results; `data/official_check` contains
  separate pixel-feature and shuffled-extraction controls.
- Not located in this tree: the per-repetition outputs and bar-chart generator behind Figure 7 (`figures/manuscript/appendix_g`
  rebuilds the final vector asset), the two-dimensional k x M sweep outputs and plotting code of Figure 5 (only the
  five `km_D*.log` files of the submission), the extraction and plotting code of the per-class decomposition of
  Figure 10 (the pooled logs hold the nine-epoch summaries), and the uniform-sphere calibration script of Appendix C
  (an external script of the submission). The additional checks in Appendix H use 10 seeds; Figure 8 uses 50 seeds.
- Checkpoints and feature files: no public download location yet.
- The authors still need to choose `LICENSE`; `CITATION.cff` includes the paper's authors and affiliations, with
  repository and archive identifiers to be added once available. `third_party/SupContrast` is the code of
  github.com/HobbitLong/SupContrast (BSD 2-Clause), see `third_party/SupContrast/THIRD_PARTY.md`.
