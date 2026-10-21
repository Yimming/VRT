# Validation, 21 October 2026

- 81 regression tests passed after removal of the old script directories.
- All 784 onset records matched `onsets_final.csv` field for field.
- The 5,061,086 result records passed schema validation; all 5,156 manifest file hashes matched.
- Fixed-input, fixed-seed comparisons preserved the original VRT, baseline,
  model and synthetic-experiment results.
- CPU and V100 smoke jobs passed all test methods, including MMDAgg, synthetic
  experiments, multiprocessing, model forward/backward passes, one SimCLR
  training batch and feature extraction in both BatchNorm modes. A further CPU
  job covered reference-direction experiments and fixed-view contrastive loss.

The cluster smoke environment used Python 3.11.3, NumPy 1.25.1 and PyTorch 2.1.2.
These are execution checks on small inputs; full model training was not repeated.
