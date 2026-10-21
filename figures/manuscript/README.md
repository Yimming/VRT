# Appendix figure builds

From the repository root:

```bash
python -m pip install -r figures/manuscript/requirements.txt
python figures/manuscript/build_all.py
```

This runs the C, F, G and I builds independently and writes to `results/figures/appendix_c/`, `appendix_f/`, `appendix_g/` and `appendix_i/`. Use `VRT_FIGURES_OUT` to select another output base. The complete script/output/input map is in [the figure guide](../README.md).

- C reads 25 exact rejection counts, each based on 500 repetitions. The separate pixel-feature and shuffled-extraction controls are not its source.
- F reads unsmoothed losses from one trained model per panel and separately validates the 50 analysis seeds used by VRT. SimCLR losses are converted from the stored SupConLoss units to NT-Xent nats by multiplying by `0.07 / 0.5`. The supervised loader uses the shared inputs in `figures/`; the SimCLR loss loader locates the released data through `results.family_path`.
- G preserves the saved vector bars and applies the saved label corrections. The original numerical repetitions are not bundled. To rerun the experiment, use `python synthetic.py reference --out results/reference.csv`.
- I reads the audited 24-model onset snapshot. It displays eight model groups and checks the 19 paired detections: nine earlier, nine equal and one later for MMD on scores versus VRT.

Fonts are not bundled. Matplotlib uses the installed serif font fallback; G retains the fonts embedded in its input PDFs.
