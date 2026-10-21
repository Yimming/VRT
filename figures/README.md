# Figure reproduction

Run from the repository root with NumPy, Matplotlib and pypdf installed:

```bash
python -m pip install -r figures/manuscript/requirements.txt
export MPLBACKEND=Agg
python figures/plot_fig3_s50.py
python figures/plot_fig4_s50.py
python figures/plot_dc_trajectory_s50.py
python figures/manuscript/build_all.py
```

PDFs and PNG previews go to `results/figures/`, which is ignored by Git. Set `VRT_FIGURES_OUT` to use another output directory. Saved CSV inputs are only read. Historical filenames retain their earlier figure numbers.

| Paper figure | Script under `figures/` | Output under `results/figures/` | Inputs |
| --- | --- | --- | --- |
| Figure 2, regularization | [plot_fig3_s50.py](plot_fig3_s50.py) | `fig3_regularization.pdf` | `vrt_abl*.csv` in [fig_data_s50/](fig_data_s50/) |
| Figure 3, synthetic sensitivity and BatchNorm | [plot_fig4_s50.py](plot_fig4_s50.py) | `fig4_diagnosis.pdf` | [fig4a_clusters.csv](fig_data_s50/fig4a_clusters.csv), [fig4a_matched.csv](fig_data_s50/fig4a_matched.csv), [fig4b_r50.csv](fig_data_s50/fig4b_r50.csv) |
| Figure 8, reference-set score trajectories | [plot_dc_trajectory_s50.py](plot_dc_trajectory_s50.py) | `fig_dc_trajectory.pdf` | `dc_*.csv` in [fig_data_s50/](fig_data_s50/), [onsets.csv](fig_data_s50/onsets.csv) |
| Appendix C, original CIFAR split | [manuscript/appendix_c/plot_official_paper.py](manuscript/appendix_c/plot_official_paper.py) | `appendix_c/fig_cifar_official_control.pdf` | [official_counts.csv](manuscript/appendix_c/official_counts.csv) |
| Figure 6, Appendix F, ten loss panels | [manuscript/appendix_f/build_figure.py](manuscript/appendix_f/build_figure.py) | `appendix_f/fig_trajectories_appendix.pdf` | [benchmark_final/](benchmark_final/), [fig_data_s50/](fig_data_s50/), [filtered SimCLR rows](manuscript/appendix_f/data/ssl/), and [SimCLR losses](../data/08_per_model_trajectories/records/simclr_loss/) |
| Appendix G, synthetic reference choice | [manuscript/appendix_g/build_figure.py](manuscript/appendix_g/build_figure.py) | `appendix_g/fig_ref_asymmetry.pdf` | Saved [source_before.pdf](manuscript/appendix_g/source_before.pdf) and [label_overlay.pdf](manuscript/appendix_g/label_overlay.pdf); see limitation below |
| Appendix I, MMD on scores | [manuscript/appendix_i/plot_horizontal.py](manuscript/appendix_i/plot_horizontal.py) | `appendix_i/fig_mmd_on_chat.pdf` | [onsets.json](manuscript/appendix_i/onsets.json) |

[manuscript/build_all.py](manuscript/build_all.py) runs the four appendix builds. Each can also be run separately. Fonts use Times New Roman when installed, with serif fallbacks; font substitution can change text spacing.

## Intermediate summaries

Regenerate the small figure-input CSVs from the released per-seed results into a **new** directory:

```bash
python figures/export_fig_data.py data results/figure_data
python figures/appD_numbers.py
```

The exporter uses `results.family_path` to locate the paper-section data folders. It refuses an existing output directory and does not replace `fig_data_s50/`. `appD_numbers.py` prints the synthetic rejection counts and bandwidth summaries, using the saved matched repetitions and [Appendix D records](../data/06_synthetic_power_profile_details/records/).

The saved `base_*.csv` snapshots predate the final MMDAgg checkpoint additions. Re-exporting adds 127 values in previously empty `mmdagg_frac` fields across six files; existing nonempty values are unchanged. Current manuscript plots do not use those fields. Use `results.py` for the final table onsets.

## Scope

Appendix C uses exact saved aggregate counts checked against the per-repetition results. Appendix F validates checkpoint and seed coverage and recomputes its VRT onsets; it applies the saved SimCLR loss conversion once. Appendix I retains all 24 model records and checks the reported comparison counts.

Appendix G reconstructs the final vector figure from its saved bars and label overlay. The original 500-repetition output and bar-chart generator are unavailable, so this is not a regeneration of the bars from numerical records. To rerun the experiment, use `python synthetic.py reference --out results/reference.csv`.

The original full two-dimensional sweep inputs for Appendix E and the original plotting generator for Appendix J are not included. These scripts do not claim to rebuild every figure in the paper.
