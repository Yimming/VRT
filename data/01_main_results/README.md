# 4.2 Main results; Tables 1–2

Main comparisons on normalized embeddings, in both BatchNorm extraction modes. Source records also support Sections 4.3–4.4 and Appendices E, G and H.

Summary CSVs: [all_onsets.csv](all_onsets.csv), [main_results.csv](main_results.csv), [batchnorm_comparison.csv](batchnorm_comparison.csv).

Individual test results, before onset aggregation:

| Method | Per-seed records |
| --- | --- |
| VRT | [records/vrt/](records/vrt/) |
| MMD | [records/mmd/](records/mmd/) |
| Energy Distance | [records/energy_distance/](records/energy_distance/) |
| C2ST | [records/c2st/](records/c2st/) |
| MMDAgg | [records/mmdagg/](records/mmdagg/) |
| SimCLR experiments | [records/simclr/](records/simclr/) |

For example, this [ResNet-50 VRT file](records/vrt/vrt_fwd_r50_bntrain_large_c0-83.csv) lists `epoch`, `seed`, score difference `dc` and `p`. Seeds repeat the test on fixed checkpoint features. MMDAgg records binary rejection decisions rather than p-values.

Runner: `evaluate.py test`. Source files in this folder: 3,576.

[Filename and column guide](../README.md#names-used-in-the-files) explains `bntrain`, `raw`, budgets, checkpoints and seeds.
