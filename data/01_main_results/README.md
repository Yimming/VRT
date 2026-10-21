# 4.2 Main results; Tables 1–2

Main comparisons on normalized embeddings, in both BatchNorm extraction modes. Source records also support Sections 4.3–4.4 and Appendices E, G and H.

| Paper result | Summary CSV |
| --- | --- |
| Table 1: eval-mode detection onsets | [main_results.csv](main_results.csv) |
| Table 2: eval/train BatchNorm comparison | [batchnorm_comparison.csv](batchnorm_comparison.csv) |
| All 784 onset cells, with checkpoint coverage | [all_onsets.csv](all_onsets.csv) |

The CSVs retain numerical onsets. In Table 1's ImageNet rows, `BN` marks detection attributed to BatchNorm-driven feature shifts; `*` marks null train-as-reference VRT with test-as-reference rejection (Appendix H).

Start with the [model-to-method index](records/) to find individual test results. Within each method, full model names, extraction modes and sample sizes identify the setting.

Individual test results, before onset aggregation:

| Method | Per-seed records |
| --- | --- |
| VRT | [records/vrt/](records/vrt/) |
| MMD | [records/mmd/](records/mmd/) |
| Energy Distance | [records/energy_distance/](records/energy_distance/) |
| C2ST | [records/c2st/](records/c2st/) |
| MMDAgg | [records/mmdagg/](records/mmdagg/) |
| SimCLR experiments | [records/simclr/](records/simclr/) |

For the ResNet-50 VRT onset at epoch 61 in Table 1, open this [eval-mode CSV](records/vrt/resnet50_cifar100/eval_mode/points_per_split_10000/vrt_fwd_r50_large_c42-83.csv) and read `epoch`, `seed` and `p`. Seeds repeat the test on fixed checkpoint features. MMDAgg records binary rejection decisions rather than p-values.

Runner: `evaluate.py test`. Source files in this folder: 3,576.

[Filename and column guide](../README.md#names-used-in-the-files) explains `bntrain`, `raw`, budgets, checkpoints and seeds.
