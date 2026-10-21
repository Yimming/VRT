# Final manuscript figures

These four packages supplement the earlier figure scripts. Run with Python 3.10 or later and NumPy, Matplotlib and pypdf installed. Paths are relative to each script, so the working directory does not matter.

From the release root:

```sh
python -m pip install -r figures/manuscript/requirements.txt
python figures/manuscript/build_all.py
```

Or from this folder:

```sh
python build_all.py
```

Each package writes its final PDF under its own `generated/` folder. C, F and I also write PNG previews. The plotting code uses Times New Roman where installed, with Liberation Serif and DejaVu Serif fallbacks. Fonts are not bundled. Font substitution can change text metrics but not data or plot geometry. G retains the fonts embedded in its saved vector inputs.

| Paper asset | Command from this folder | Data and checks |
|---|---|---|
| `fig_cifar_official_control.pdf`, Appendix C | `python appendix_c/plot_official_paper.py` | `official_counts.csv`, 25 exact rejection counts over 500 repetitions each, originally parsed from the saved official-split summary and now verified against the `official` rows of `data/null_cifar_vrt` and `data/null_cifar_base`. C2ST uses p < 0.05 without the accuracy threshold. |
| `fig_trajectories_appendix.pdf`, Appendix F | `python appendix_f/build_figure.py` | Eight supervised loss records, their 50-seed median-p trajectories, and two complete SimCLR loss and per-seed VRT trajectories. The two loader modules validate epoch coverage, seed coverage and onset. |
| `fig_ref_asymmetry.pdf`, Appendix G | `python appendix_g/build_figure.py` | Saved original vector figure plus the saved final label overlay. See the reproducibility limitation below. |
| `fig_mmd_on_chat.pdf`, Appendix I | `python appendix_i/plot_horizontal.py` | `onsets.json`, all 24 model records. Eight displayed groups cover 16 individual models; zoo bands preserve the minimum and maximum of the five model onsets. |

`source_manifest.json` gives the source hashes and the transformation used for each packaged input. Source locations there are logical archive locations, not machine-specific absolute paths. The scripts are portable adaptations of the final manuscript-side generators, preserving calculations, panel positions, labels and colours.

## Appendix F

The final figure has ten loss panels. The old `plot_trajectories_s50.py` in the parent folder is an earlier plot and is not a substitute for this build.

Only existing VRT rows with `stat=vrt` and `budget=b10000` were retained from the SimCLR fast-result files, with the five fields consumed by the loader. They are concatenated and sorted, without recalculating p-values. Each dataset has exactly 1,000 checkpoints and seeds 42 to 91 at every checkpoint. Median-p onsets are 229 and 191. The supervised onsets are 38, 61, no detection, 34, 33, no detection, 35 and no detection in plotted order.

Loss curves are unsmoothed checkpoint measurements from one trained model, not medians over the 50 VRT seeds. SimCLR uses the eval-mode loss fields and multiplies the recorded SupConLoss values by 0.07 / 0.5 to express NT-Xent loss in nats. Epoch 0 is omitted only from the supervised loss display. The loaders keep this conversion separate and the final build applies it once.

## Appendix I

The compact input retains only fields used by the final plot. The underlying onset values are unchanged from its audited manuscript input. Earlier snapshot coverage diagnostics were not copied into the plot input, because they are not figure data and do not describe the final release coverage. Both tests detect divergence in 19 models. MMD on c-hat is earlier in nine, equal in nine and later in one, with an absolute onset difference at most 14 epochs. The code asserts these counts. This package does not replace the release tools that recompute onsets from per-seed results.

## Remaining source-data limitations

Appendix C is reproducible from exact saved aggregate counts, now verified against all 500 repetitions per setting in `data/null_cifar_vrt` and `data/null_cifar_base` (the `official` split). `official_counts.csv` remains an aggregate table. The `data/official_check` files hold separate pixel-feature and shuffled-extraction controls and do not supply the plotted counts.

Appendix G is a reconstruction of the final vector asset, not a reconstruction from raw experimental output. The original `source_before.pdf` contains all bars and displayed values. `build_figure.py` makes the two final label corrections and merges the saved `label_overlay.pdf`; it changes no bar height or result. The original 500-repetition output and the original bar-chart generator were not found locally. The as-run experimental script `synthetic_ref_asymmetry.py` is included as background and is **not** called by `build_all.py`; running it would launch the full experiment. No exact counts have been inferred from the rounded percentage labels. The labels in the experimental script still use the old shorthand `dk=0.3`, while its arithmetic correctly uses `0.3 * sqrt(D)`.
