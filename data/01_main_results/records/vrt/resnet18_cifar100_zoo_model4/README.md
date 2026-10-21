# ResNet-18 / CIFAR-100 zoo: model 4

VRT.

The CSV `p` column contains per-test p-values.

Ranges below come from the CSV rows; the number in parentheses is the number of distinct checkpoints or seeds.

## Eval extraction (running BatchNorm statistics) · 10,000 points per split

| File | Checkpoints | Test seeds |
| --- | --- | --- |
| [vrt_fwd_zoo_c100_m4_large_c0-41.csv](eval_mode/points_per_split_10000/vrt_fwd_zoo_c100_m4_large_c0-41.csv) | 0–41 (42) | 42–91 (50) |
| [vrt_fwd_zoo_c100_m4_large_c42-60.csv](eval_mode/points_per_split_10000/vrt_fwd_zoo_c100_m4_large_c42-60.csv) | 42–60 (19) | 42–91 (50) |

## Eval extraction (running BatchNorm statistics) · 1,500 points per split

| File | Checkpoints | Test seeds |
| --- | --- | --- |
| [vrt_fwd_zoo_c100_m4_small_c0-41.csv](eval_mode/points_per_split_1500/vrt_fwd_zoo_c100_m4_small_c0-41.csv) | 0–41 (42) | 42–91 (50) |
| [vrt_fwd_zoo_c100_m4_small_c42-60.csv](eval_mode/points_per_split_1500/vrt_fwd_zoo_c100_m4_small_c42-60.csv) | 42–60 (19) | 42–91 (50) |

## Train extraction (current-batch BatchNorm statistics) · 10,000 points per split

| File | Checkpoints | Test seeds |
| --- | --- | --- |
| [vrt_fwd_zoo_c100_m4_bntrain_large_c0-60.csv](train_mode/points_per_split_10000/vrt_fwd_zoo_c100_m4_bntrain_large_c0-60.csv) | 0–60 (61) | 42–91 (50) |

## Train extraction (current-batch BatchNorm statistics) · 1,500 points per split

| File | Checkpoints | Test seeds |
| --- | --- | --- |
| [vrt_fwd_zoo_c100_m4_bntrain_small_c0-60.csv](train_mode/points_per_split_1500/vrt_fwd_zoo_c100_m4_bntrain_small_c0-60.csv) | 0–60 (61) | 42–91 (50) |
