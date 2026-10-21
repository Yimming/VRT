# ResNet-50 / CIFAR-100

MMDAgg.

`p` records the MMDAgg rejection decision (0 or 1). It is not a p-value.

Ranges below come from the CSV rows; the number in parentheses is the number of distinct checkpoints or seeds.

## Eval extraction (running BatchNorm statistics) · 1,500 points per split

| File | Checkpoints | Test seeds |
| --- | --- | --- |
| [mmdagg_r50_b1500_c0-15.csv](eval_mode/points_per_split_1500/mmdagg_r50_b1500_c0-15.csv) | 0–15 (16) | 42–91 (50) |
| [mmdagg_r50_b1500_c16-31.csv](eval_mode/points_per_split_1500/mmdagg_r50_b1500_c16-31.csv) | 16–31 (16) | 42–91 (50) |
| [mmdagg_r50_b1500_c32-47.csv](eval_mode/points_per_split_1500/mmdagg_r50_b1500_c32-47.csv) | 32–47 (16) | 42–91 (50) |
| [mmdagg_r50_b1500_c48-63.csv](eval_mode/points_per_split_1500/mmdagg_r50_b1500_c48-63.csv) | 48–63 (16) | 42–91 (50) |
| [mmdagg_r50_b1500_c64-79.csv](eval_mode/points_per_split_1500/mmdagg_r50_b1500_c64-79.csv) | 64–79 (16) | 42–91 (50) |
| [mmdagg_r50_b1500_c80-95.csv](eval_mode/points_per_split_1500/mmdagg_r50_b1500_c80-95.csv) | 80–95 (16) | 42–91 (50) |
| [mmdagg_r50_b1500_c96-111.csv](eval_mode/points_per_split_1500/mmdagg_r50_b1500_c96-111.csv) | 96–111 (16) | 42–91 (50) |
| [mmdagg_r50_b1500_c112-127.csv](eval_mode/points_per_split_1500/mmdagg_r50_b1500_c112-127.csv) | 112–127 (16) | 42–91 (50) |
| [mmdagg_r50_b1500_c128-143.csv](eval_mode/points_per_split_1500/mmdagg_r50_b1500_c128-143.csv) | 128–143 (16) | 42–91 (50) |
| [mmdagg_r50_b1500_c144-159.csv](eval_mode/points_per_split_1500/mmdagg_r50_b1500_c144-159.csv) | 144–159 (16) | 42–91 (50) |
| [mmdagg_r50_b1500_c160-175.csv](eval_mode/points_per_split_1500/mmdagg_r50_b1500_c160-175.csv) | 160–175 (16) | 42–91 (50) |
| [mmdagg_r50_b1500_c176-191.csv](eval_mode/points_per_split_1500/mmdagg_r50_b1500_c176-191.csv) | 176–191 (16) | 42–91 (50) |
| [mmdagg_r50_b1500_c192-200.csv](eval_mode/points_per_split_1500/mmdagg_r50_b1500_c192-200.csv) | 192–200 (8) | 42–91 (50) |

## Eval extraction (running BatchNorm statistics) · 5,000 points per split

[Open 124 CSV files with checkpoint and seed ranges](eval_mode/points_per_split_5000).

## Train extraction (current-batch BatchNorm statistics) · 1,500 points per split

| File | Checkpoints | Test seeds |
| --- | --- | --- |
| [mmdagg_r50_bntrain_b1500_c0-24.csv](train_mode/points_per_split_1500/mmdagg_r50_bntrain_b1500_c0-24.csv) | 0–24 (25) | 42–91 (50) |
| [mmdagg_r50_bntrain_b1500_c25-49.csv](train_mode/points_per_split_1500/mmdagg_r50_bntrain_b1500_c25-49.csv) | 25–49 (25) | 42–91 (50) |
| [mmdagg_r50_bntrain_b1500_c50-74.csv](train_mode/points_per_split_1500/mmdagg_r50_bntrain_b1500_c50-74.csv) | 50–74 (25) | 42–91 (50) |
| [mmdagg_r50_bntrain_b1500_c75-99.csv](train_mode/points_per_split_1500/mmdagg_r50_bntrain_b1500_c75-99.csv) | 75–99 (25) | 42–91 (50) |
| [mmdagg_r50_bntrain_b1500_c100-124.csv](train_mode/points_per_split_1500/mmdagg_r50_bntrain_b1500_c100-124.csv) | 100–124 (25) | 42–91 (50) |
| [mmdagg_r50_bntrain_b1500_c125-149.csv](train_mode/points_per_split_1500/mmdagg_r50_bntrain_b1500_c125-149.csv) | 125–149 (25) | 42–91 (50) |
| [mmdagg_r50_bntrain_b1500_c150-174.csv](train_mode/points_per_split_1500/mmdagg_r50_bntrain_b1500_c150-174.csv) | 150–174 (25) | 42–91 (50) |
| [mmdagg_r50_bntrain_b1500_c175-200.csv](train_mode/points_per_split_1500/mmdagg_r50_bntrain_b1500_c175-200.csv) | 175–200 (25) | 42–91 (50) |

## Train extraction (current-batch BatchNorm statistics) · 5,000 points per split

[Open 105 CSV files with checkpoint and seed ranges](train_mode/points_per_split_5000).
