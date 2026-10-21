# ResNet-50 / CIFAR-100

VRT.

The CSV `p` column contains per-test p-values.

Ranges below come from the CSV rows; the number in parentheses is the number of distinct checkpoints or seeds.

## Eval extraction (running BatchNorm statistics) · 10,000 points per split

| File | Checkpoints | Test seeds |
| --- | --- | --- |
| [vrt_fwd_r50_large_c0-41.csv](eval_mode/points_per_split_10000/vrt_fwd_r50_large_c0-41.csv) | 0–41 (42) | 42–91 (50) |
| [vrt_fwd_r50_large_c42-83.csv](eval_mode/points_per_split_10000/vrt_fwd_r50_large_c42-83.csv) | 42–83 (42) | 42–91 (50) |
| [vrt_fwd_r50_large_c84-125.csv](eval_mode/points_per_split_10000/vrt_fwd_r50_large_c84-125.csv) | 84–125 (42) | 42–91 (50) |
| [vrt_fwd_r50_large_c126-167.csv](eval_mode/points_per_split_10000/vrt_fwd_r50_large_c126-167.csv) | 126–167 (42) | 42–91 (50) |
| [vrt_fwd_r50_large_c168-200.csv](eval_mode/points_per_split_10000/vrt_fwd_r50_large_c168-200.csv) | 168–200 (32) | 42–91 (50) |

## Eval extraction (running BatchNorm statistics) · 1,500 points per split

| File | Checkpoints | Test seeds |
| --- | --- | --- |
| [vrt_fwd_r50_small_c0-41.csv](eval_mode/points_per_split_1500/vrt_fwd_r50_small_c0-41.csv) | 0–41 (42) | 42–91 (50) |
| [vrt_fwd_r50_small_c42-83.csv](eval_mode/points_per_split_1500/vrt_fwd_r50_small_c42-83.csv) | 42–83 (42) | 42–91 (50) |
| [vrt_fwd_r50_small_c84-125.csv](eval_mode/points_per_split_1500/vrt_fwd_r50_small_c84-125.csv) | 84–125 (42) | 42–91 (50) |
| [vrt_fwd_r50_small_c126-167.csv](eval_mode/points_per_split_1500/vrt_fwd_r50_small_c126-167.csv) | 126–167 (42) | 42–91 (50) |
| [vrt_fwd_r50_small_c168-200.csv](eval_mode/points_per_split_1500/vrt_fwd_r50_small_c168-200.csv) | 168–200 (32) | 42–91 (50) |

## Train extraction (current-batch BatchNorm statistics) · 10,000 points per split

| File | Checkpoints | Test seeds |
| --- | --- | --- |
| [vrt_fwd_r50_bntrain_large_c0-83.csv](train_mode/points_per_split_10000/vrt_fwd_r50_bntrain_large_c0-83.csv) | 0–83 (84) | 42–91 (50) |
| [vrt_fwd_r50_bntrain_large_c84-167.csv](train_mode/points_per_split_10000/vrt_fwd_r50_bntrain_large_c84-167.csv) | 84–167 (84) | 42–91 (50) |
| [vrt_fwd_r50_bntrain_large_c168-200.csv](train_mode/points_per_split_10000/vrt_fwd_r50_bntrain_large_c168-200.csv) | 168–200 (32) | 42–91 (50) |

## Train extraction (current-batch BatchNorm statistics) · 1,500 points per split

| File | Checkpoints | Test seeds |
| --- | --- | --- |
| [vrt_fwd_r50_bntrain_small_c0-83.csv](train_mode/points_per_split_1500/vrt_fwd_r50_bntrain_small_c0-83.csv) | 0–83 (84) | 42–91 (50) |
| [vrt_fwd_r50_bntrain_small_c84-167.csv](train_mode/points_per_split_1500/vrt_fwd_r50_bntrain_small_c84-167.csv) | 84–167 (84) | 42–91 (50) |
| [vrt_fwd_r50_bntrain_small_c168-200.csv](train_mode/points_per_split_1500/vrt_fwd_r50_bntrain_small_c168-200.csv) | 168–200 (32) | 42–91 (50) |
