# ResNet-18 / CIFAR-10: C, strong augmentation

MMD on unnormalized features.

The CSV `p` column contains per-test p-values.

Ranges below come from the CSV rows; the number in parentheses is the number of distinct checkpoints or seeds.

## Eval extraction (running BatchNorm statistics) · 10,000 points per split

| File | Checkpoints | Test seeds |
| --- | --- | --- |
| [mmd_ablC_n10000_c0-39.csv](eval_mode/points_per_split_10000/mmd_ablC_n10000_c0-39.csv) | 0–39 (40) | 42–91 (50) |
| [mmd_ablC_n10000_c40-49.csv](eval_mode/points_per_split_10000/mmd_ablC_n10000_c40-49.csv) | 40–49 (10) | 42–91 (50) |

## Eval extraction (running BatchNorm statistics) · 1,500 points per split

| File | Checkpoints | Test seeds |
| --- | --- | --- |
| [mmd_ablC_n1500_c0-39.csv](eval_mode/points_per_split_1500/mmd_ablC_n1500_c0-39.csv) | 0–39 (40) | 42–91 (50) |
| [mmd_ablC_n1500_c40-49.csv](eval_mode/points_per_split_1500/mmd_ablC_n1500_c40-49.csv) | 40–49 (10) | 42–91 (50) |

## Train extraction (current-batch BatchNorm statistics) · 10,000 points per split

| File | Checkpoints | Test seeds |
| --- | --- | --- |
| [mmd_ablC_bntrain_n10000_c0-39.csv](train_mode/points_per_split_10000/mmd_ablC_bntrain_n10000_c0-39.csv) | 0–39 (40) | 42–91 (50) |
| [mmd_ablC_bntrain_n10000_c40-49.csv](train_mode/points_per_split_10000/mmd_ablC_bntrain_n10000_c40-49.csv) | 40–49 (10) | 42–91 (50) |

## Train extraction (current-batch BatchNorm statistics) · 1,500 points per split

| File | Checkpoints | Test seeds |
| --- | --- | --- |
| [mmd_ablC_bntrain_n1500_c0-39.csv](train_mode/points_per_split_1500/mmd_ablC_bntrain_n1500_c0-39.csv) | 0–39 (40) | 42–91 (50) |
| [mmd_ablC_bntrain_n1500_c40-49.csv](train_mode/points_per_split_1500/mmd_ablC_bntrain_n1500_c40-49.csv) | 40–49 (10) | 42–91 (50) |
