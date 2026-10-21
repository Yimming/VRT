# Release checks, October 18 (two rounds)

What was verified while assembling this repository, and what the second review (manuscript side, three rounds of
checks on the first version of this tree) found and how it was addressed. `table_comparison_1017.csv` is the
manuscript-side comparison of every value of Tables 1 to 3 with the data; `patches_vs_asrun_1018.diff` lists every
change relative to the as-run scripts.

## Round 1 (assembly)

Code. The 101 scripts of the cluster code folder (snapshot of October 18) compile; the runners and libraries patched
for the release keep their test arithmetic (the reviewer's independent differencing agreed with the diff line by line
and 22 pairs of old/new return values and recorded random draws were identical over float32/float64, MMD, ED, VRT full,
small and reversed, and C2ST).

Data export (`tools/export_results.py --dry_run` on the complete copy of the result folders): 4,421 files, no key with
two different values; 11 files with NUL regions (7 `mmdagg/mmdagg_r50_b5000_c1xx-1xx.csv`,
`mmdagg/mmdagg_zoo_c100_m3_b5000_c54-54.csv`, 3 `ssl_baselines/sslagg_cifar100_s42-91_e{525,625,675}.csv`), each leaving
exactly 50 intact rows per epoch; 1 file with 50 verbatim duplicate rows (`mmdagg/mmdagg_r50_b5000_c149-149.csv`); no
torn lines in the as-run data (5,092,464 rows); `p_exact` added to 92 `reversed/` files and `p_henze_exact` to 168
`knn_v2/` files. The reviewer's independent export reproduced every SHA256 of the manifest.

Onsets (`tools/make_onsets.py` on the same copy): 768 cells; every entry of Tables 1, 2 and 3 of the manuscript
(`Table/table{1,2,3}.tex`) is reproduced, also by the reviewer's independent aggregation that reads 4,506,950 unique
seed records without the release code (same onsets, coverage, gaps and seed counts in all 768 cells); no cell has an
epoch at median p = 0.05 exactly under the exact p-values (three under the stored four-decimal values: r50 train Henze
epoch 83, zoo_c10_m1 train Henze epoch 35, zoo_c10_m3 eval reversed VRT epoch 30; only the last moves an onset, 31 to
29, Appendix G); no epoch with fewer than 50 seeds; two cells with unevaluated checkpoints before their onset (SimCLR
CIFAR-10 and CIFAR-100, train mode, MMDAgg at 1,500 points: their fill was still running when this copy was taken).
Reversed-VRT onsets of the ten zoo models at the large budget (exact p): 27, 31, 34, 29, 30 (CIFAR-10 m0 to m4) and
34, 29, 29, 26, 28 (CIFAR-100), forward VRT 26, 25, 30, 29, 30 and 34, 28, 29, 26, 27; lags 1, 6, 4, 0, 0, 0, 1, 0, 0,
1, median 0.5.

Figures. The scripts in `figures/` run from the folder with relative paths; `figures/manuscript/build_all.py` rebuilds
the four appendix figures (their text and rasterised rendering match the paper's PDFs pixel for pixel at a 1,400-pixel
long edge, per the reviewer).

## Round 2 (review findings and what changed)

1. Figure 8 (`fig_dc_trajectory.pdf`) in the manuscript was still the submission's trajectory (ImageNet simple at
   epoch 300: standard -0.084, reversed 0.812 in the paper's vector paths; 50-seed medians 0.0007 and 1.046). The
   release script `plot_dc_trajectory_s50.py` draws the verified 50-seed data; the manuscript figure is to be replaced
   with its output (manuscript task). The four `dc_<tag>.csv` files were checked by the reviewer against the per-seed
   files, 418 epochs, both directions.
2. Active VRT paths could turn invalid statistics into the minimum p (`supcon_fill_vrt.vrt_test`, used by the SimCLR
   runner, had no checks; `check_scores` allowed +inf but no check covered a NaN or infinite median difference).
   Fixed: `check_scores` and the median-statistic guards in `vrt_low_lib.vrt_low` / `vrt_full`,
   `supcon_fill_vrt.vrt_test` and `vrt_reversed_budget.perm_p`; the `compute_chat` docstring now states that duplicate
   reference points give NaN scores; test `test_vrt_paths_stop_on_nan_scores_and_non_finite_statistics`. No random
   draw changed. Evidence from the data: the recorded SimCLR MMD-on-c-hat runs have at most 2 +inf scores per draw
   (CIFAR-100, 865 checkpoints; CIFAR-10 none), all stored `dc` values are finite; the historical permutation
   statistics themselves were not logged, so this is a guard for future runs, not a proof about every past draw.
3. `ResumeIndex` accepted torn or invalid rows as complete and rows of other runs. Fixed: rows count only when they
   belong to the current run (`expect`), parse, and carry valid values (`ranges`, `nan_ok`); invalid rows and
   conflicts make `report()` refuse; test `test_resume_index_counts_only_valid_rows_of_this_run`.
4. `open_append` could attach new columns to an old header or new rows to a torn last line. Fixed: header must equal
   the runner's schema, torn tails are moved to a `.torn` sidecar (`repair_tail`), also done by `ResumeIndex`; tests
   `test_resume_index_and_open_append_refuse_a_foreign_header`, `test_torn_last_line_is_isolated_before_appending`.
5. `--raw` entry points refused zero rows. Fixed with `allow_zero_rows=args.raw` in `mmd_tst_fast.py`,
   `energy_tst_fast.py`, `knn_tst_budget.py`, `base_fill.py`; test `test_raw_entry_points_accept_zero_rows`.
6. The data tools did not validate their inputs (cross-file conflicts, rows foreign to their file, NaN p read as
   "not rejected", MMDAgg decision 2, unparsable rows skipped, re-export into a non-empty folder, arbitrary numbers
   projected onto the p lattice). Fixed: `tools/result_schemas.py` (per-family schemas), export validates and refuses,
   requires an empty destination, removes a half-written destination on refusal, marks legacy files; `make_onsets.py`
   validates before aggregating and refuses invalid decisions; `exact_perm_p` refuses off-lattice values; tests in
   `test_export_results.py`, `test_make_onsets.py`, `test_release_guards.py`.
7. Two documented entry points failed from the tree (`tools/tests_timing.py` could not import the runners;
   `figures/appD_numbers.py` lacked `mmd_bandwidth.csv`). Fixed: the tools put `code/` on the path;
   `appD_numbers.py` takes `--bandwidth` and reports missing inputs; `mmd_bandwidth.csv` lives in `data/synth` after the
   export (it is on the cluster, `s50/synth/`, not yet in the local copy).
8. README over-statements (scope, protocol paragraph, completeness, "every runner resumes", Appendix G sync,
   "no checkpoint from a resumed run") rewritten; the protocol paragraph follows the reviewer's wording.
9. ENVIRONMENT.md: RNG statement corrected (RandomState for VRT, MMD, ED, C2ST, MMDAgg; default_rng for kNN/Henze),
   GPU jobs ran on V100 or A100 per the logs, threads 8 only for CPU jobs and the timing run, pip freeze marked as
   archival, pypdf for the manuscript packages.
10. Figure and table provenance rewritten with the camera-ready numbering (`figures/README.md`, README table),
    including the four manuscript packages and the sources that are not in the tree.
11. LEGACY headers added to the earlier-stage helpers; nothing moved or deleted.

After the round-2 changes the validating tools were run on the complete local copy of the results: the export dry run
accepts all 4,421 files (4,887,676 validated records, 68 legacy files, no conflicting key, no row foreign to its file,
no invalid value; two naming variants, `_c<epoch>` chunks and `sslagg_*_s52-91_*` seed files, were added to the
schemas when the first run flagged them; a Mac-side scratch file `mmdagg5k_fractions_1009.csv` at the top level is the
only unrecognised name and must be removed from the source folder or passed with `--allow_unknown`), and
`make_onsets.py` with validation reproduces the 768 onsets of round 1 exactly (4,507,000 records in the families it
reads).

Test suite after round 2: 21 tests (`python -m pytest tests -q`), after round 3: 27, with the Type-I schemas: 28, after the fourth review: 30. The `supcon_fill_vrt.vrt_test` guard assertion runs
only where torch is installed (the module imports it); the same guard code is exercised through `vrt_full`.

## Round 3 (second review of October 18, findings 2, 3, 6, 7 and 9)

1. Resume did not see a changed sample budget, permutation count or normalization (finding 2). The runners now put
   the full configuration into the resume check (`tag`, `n_samples`, `n_perm`, `features`, and `k` / `M` / `direction`
   where they apply), refuse rows of another run instead of ignoring them (one output file holds one run), refuse a
   budget larger than a split instead of testing `min(budget, split)` silently, and record the normalization in a
   `features` column (`l2` / `raw`); tests `test_resume_index_counts_only_valid_rows_of_this_run`,
   `test_runners_refuse_a_budget_larger_than_the_split`.
2. Resume accepted an MMDAgg decision of 0.5 and inconsistent or missing permutation counts (finding 3). `ResumeIndex`
   takes `discrete` value sets and a `row_check`; `count_consistent` requires B >= 1, 0 <= count <= B and
   p == (count + 1)/(B + 1) within six-decimal rounding; test `test_resume_index_checks_permutation_counts`.
3. Data tools (finding 6): the schemas parse integer keys (`'042'` is `42`, `'oops'` is an error), check the count
   columns and an existing exact-p column against p (`cross` rules), carry the fixed protocol of the runs
   (`PROTOCOL`: direction, M, k, n_perm per family and budget) which `make_onsets.py` enforces by default
   (`--no_protocol_check` to relax), and `make_onsets.py` takes the exact p from the count columns when present and
   recovers it from the four-decimal value only for as-run files; a cell whose epochs all lack a seed is written with
   the status "no complete epoch" and listed, never dropped; tests in `test_data_tool_edges.py`.
4. Export cleanup (finding 7): the export writes to a partial folder beside the destination and moves it into place
   when complete; a refusal removes only that folder, an empty pre-existing destination stays untouched; an empty
   source stops with a message; test `test_export_uses_a_partial_folder_and_refuses_an_empty_source`.
5. Wording (finding 9): README (Appendix C for the uniform-sphere script, Appendix H 10 seeds vs Figure 8 50 seeds,
   +inf statement limited to finite median differences, "unchanged random stream" limited to valid inputs of the
   statistical tests), ENVIRONMENT (no "no onset changed", OMP settings per job script, RandomState(1000 + seed)
   limited to the reversed budget draw, float32 limited to the main-budget SimCLR VRT), `tools/tests_timing.py`
   (CPU/GPU neighbour differences, float64 scope, command from the release root), `figures/appD_numbers.py`
   (percentages with one decimal), this file (7 r50 NUL files).

After round 3 the validating tools were run again on the complete local copy (the Mac-side scratch file moved out
by the reviewer): export dry run 4,420 files, 4,887,676 validated records, 0 conflicts, protocol values satisfied by
every row; `make_onsets.py` with protocol checks gives the same 768 cells and onset fields as round 1 (status: 432
final, 334 null with the last checkpoint evaluated, 2 provisional, the two pending SimCLR train-mode MMDAgg cells).
Test suite: 27 tests.

## Final data (October 19)

The last fill job (32910000, SimCLR train-mode MMDAgg at 1,500 points) completed; `check_progress.sh` reports both
cells FINAL (onsets 801 and 679, no gap, every later checkpoint evaluated) and the float64 VRT timing job (32915311)
completed. The complete result folders were pulled from the cluster (`ssl_bntrain`, `bntrain_mmdagg`, `bntrain_c2st`,
`timing`, `synth`, `null`, `null_trtr`, `null_cifar_base`, `null_cifar_rule`, `null_cifar_vrt`, `official_check`,
`km_D*.log`); schemas for the Type-I families were added (`null`/`null_trtr` mirror the main families in subfolders).
`tools/export_results.py` exported 5,156 files (5,061,086 validated records, 68 legacy files, 21 non-CSV files; 11
files with NUL regions, 50 duplicate rows in one file, no torn line, no conflicting key) into `data/` (276 MB,
`data/MANIFEST.csv`). `tools/make_onsets.py --data data` (`docs/onsets_final.csv`, `docs/onsets_final_stdout.txt`):
784 cells, 435 final and 349 null with the last checkpoint evaluated, none provisional, none with a gap before the
onset, none with a median p of exactly 0.05, none with missing seeds. Checked against the manuscript: Tables 1 to 3
(as before), the kNN table (Schilling, Henze, standard and reversed VRT at 10,000 points), the small-budget table
(1,500 points, eval mode) and the reversed-VRT table agree with the final data entry by entry. The train-mode
small-budget SimCLR cells, which are not in the paper, are VRT 846/674, MMD 946/808, MMDAgg 801/679, C2ST null/null,
ED 959/815, Henze 954/902, Schilling 943/902 (CIFAR-10/CIFAR-100).

Appendix K (float64 `vrt_full`, `data/timing/tests_timing_*_vrt64.csv`, node vink26 with two AMD EPYC 7763, 8
threads, median over seeds 42 to 51): ResNet-50 epoch 61: 2.36 s at 10,000 points, 0.28 s at 1,500; ResNet20-FRN
epoch 38: 1.86 s and 0.25 s (the float32 `vrt_test` path measured 2.04 and 1.76 s on the same node; the reversed VRT
2.20/0.29 and 1.63/0.24 s). The manuscript table (1.98/1.66 and 0.27/0.23, float32 path) is to be replaced by the
float64 values; the sentence "approximately 2 seconds" stands.

Fourth review (October 19) confirmed the final data: 28 tests, 784 cells and 472 table entries of the manuscript
identical to an independent recomputation, 5,156 exported files identical in size, hash and cleaning to an independent
rebuild from the raw files, Table 5 and the Appendix C counts recomputed from the per-run files, Appendix K values
applied and compiled. Two schema defects it found were fixed in this version: `_check` tested `float(s)` before the
`FEATURES` kind, so the valid values `l2` and `raw` written by the released runners were refused (now checked before the
conversion); and a SimCLR file in the wrong folder (a `_bntrain` file under `ssl/`, or an eval-mode file under
`ssl_bntrain/`) validated but was never read by the onset loader, so `schema_for` now refuses such a file with the
folder it belongs in. Tests `test_features_column_of_the_released_runners_is_accepted` and
`test_simclr_file_in_the_wrong_folder_is_refused`; 30 tests in total. Neither defect touched the archived data (no
as-run file carries the `features` column; every SimCLR file sits in its folder).

Remaining: commit the manuscript edits (Figure 8 asset, Appendix G 0.5, checklist seeds, Appendix E "at or above",
Appendix I protocol sentence, Appendix K values), LICENSE and CITATION.cff, the GitHub repository, and the Code and
Data Availability paragraph once the URL exists. `data/` (276 MB, largest file 4 MB) fits in a GitHub repository; an
archived copy with a DOI (for example Zenodo) is the usual complement.
