# Appendix C; Table 5

Random-network calibration, disjoint-training controls and official CIFAR split checks.

[calibration.csv](calibration.csv) contains single-test rejection counts. Individual results are grouped below; the 50-seed decision-rule controls are separate from these counts.

| Control | Individual results |
| --- | --- |
| CIFAR random networks, VRT | [VRT records](records/cifar/vrt/) |
| CIFAR random networks, baselines | [Baseline records](records/cifar/baselines/) |
| CIFAR 50-seed decision rule | [Decision-rule records](records/cifar/decision_rule/) |
| CIFAR pixel and shuffled-extraction controls | [Split-check records](records/cifar/split_checks/) |
| ImageNet training/test comparison | [Training/test records](records/imagenet/train_test/) |
| ImageNet disjoint training subsets | [Disjoint-training records](records/imagenet/disjoint_training/) |

Use the [Appendix C commands](../../REPRODUCIBILITY.md#calibration-and-split-controls-appendix-c) to run new controls. To summarize the saved results:

```bash
python results.py type-i --data data --out results/type_i.csv
```

[Filename and column guide](../README.md#names-used-in-the-files) explains `bntrain`, `raw`, budgets, checkpoints and seeds.
