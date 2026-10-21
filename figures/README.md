# Figures

Scripts and data of the figures drawn from the 50-seed results. Each script in this folder reads only files inside
the folder (`fig_data_s50/`, `benchmark_final/`) and writes its PDF and PNG next to itself. The final builds of the
four appendix figures that were produced on the manuscript side are in `manuscript/` (own README and requirements,
`python manuscript/build_all.py`).

```
cd figures
python plot_fig3_s50.py            # Figure 2 of the paper (file fig3_regularization.pdf): regularization intervention, A to D
python plot_fig4_s50.py            # Figure 3 (file fig4_diagnosis.pdf): synthetic power profiles (a), ResNet-50 eval vs train mode (b)
python plot_dc_trajectory_s50.py   # Figure 8, Appendix H (fig_dc_trajectory.pdf): dc under train-as-reference and test-as-reference VRT
python plot_trajectories_s50.py    # earlier per-model loss and VRT trajectory plot (superseded by manuscript/appendix_f, kept for reference)
python plot_fig2_s50.py            # trajectory figure of ResNet20-FRN and ResNet-50 (fig2_trajectory.pdf; commented out in the camera-ready)
python appD_numbers.py --dir fig_data_s50 --bandwidth ../data/synth/mmd_bandwidth.csv   # numbers of Figure 3(a) and Appendix D
```

File names of the figures kept their submission names (`fig3_regularization.pdf` is Figure 2 of the camera-ready,
`fig4_diagnosis.pdf` is Figure 3) because `fig2_trajectory.pdf` is no longer displayed.

`fig_data_s50/` holds the per-epoch medians over the 50 seeds that the plots read (`vrt_<tag>.csv`, `base_<tag>.csv`,
`dc_<tag>.csv`, `onsets.csv`, `fig4b_r50.csv`), written from the result data by `export_fig_data.py`
(`python export_fig_data.py <data folder> fig_data_s50`), and the synthetic results `fig4a_clusters.csv`
(`code/synth_fig4a_clusters.py`) and `fig4a_matched.csv` (`code/synth_fig4a_matched.py`). `benchmark_final/` holds
the training logs with the train and test losses (unchanged from the submission). `synthetic/` holds the remaining
synthetic result files (`fig4a_all.csv` from `code/synth_fig4a_all.py`, `synth_radial_results.csv` from
`code/synth_radial_control.py`) and the task lines of those runs; `mmd_bandwidth.csv` (`code/mmd_bandwidth_control.py`)
lives in `data/synth`.

Provenance of every figure file of the camera-ready (`Figures/` of the manuscript):

| figure | file | generator | notes |
|---|---|---|---|
| Figure 1 | `vrt_pipeline.tex` | TikZ in the manuscript | schematic, no data |
| Figure 2 | `fig3_regularization.pdf` | `plot_fig3_s50.py` | VRT medians over the 50 seeds; onsets A 33, C 35 |
| Figure 3 | `fig4_diagnosis.pdf` | `plot_fig4_s50.py` | three groups at matched budget (a); ResNet-50 onsets eval vs train (b); the paper file is regenerated here |
| Figure 4, Appendix C | `fig_cifar_official_control.pdf` | `manuscript/appendix_c/plot_official_paper.py` | from the 25 aggregate rejection counts (500 repetitions each), verified against the `official` rows in `data/null_cifar_vrt` and `data/null_cifar_base` |
| Figure 5, Appendix E | `fig_km_sweep.pdf` | unchanged from the submission | the two-dimensional k x M outputs and plotting code are not in this tree; `km_D*.log` of the submission hold five one-dimensional sweeps |
| Figure 6, Appendix F | `fig_trajectories_appendix.pdf` | `manuscript/appendix_f/build_figure.py` | ten panels; loss curves are checkpoint measurements of one trained model, VRT onsets medians over 50 seeds; SimCLR losses converted once by 0.07 / 0.5 to NT-Xent nats |
| Figure 7, Appendix G | `fig_ref_asymmetry.pdf` | `manuscript/appendix_g/build_figure.py` | rebuild of the final vector asset with its label corrections; the 500-repetition outputs and the original bar-chart generator are not in this tree |
| Figure 8, Appendix H | `fig_dc_trajectory.pdf` | `plot_dc_trajectory_s50.py` | 50-seed medians of dc, standard and reversed; the paper file is being updated to this version |
| Figure 9, Appendix I | `fig_mmd_on_chat.pdf` | `manuscript/appendix_i/plot_horizontal.py` | onset bands of VRT, MMD on c-hat and embedding MMD for the 24 models (eight groups displayed); the onsets are those of `code/onc_summary.py` |
| Figure 10, Appendix J | `fig_decompose.pdf` | unchanged from the submission | per-class decomposition; the extraction and plotting code are not in this tree (the `pooled_*.log` files hold the nine-epoch summaries) |

Matplotlib 3.10 was used for the paper's files; the scripts run with any recent matplotlib. `manuscript/` needs
`pypdf` in addition.
