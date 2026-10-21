# ResNet-18 / CIFAR-100 zoo: model 1

Schilling and Henze.

`p_schilling` and `p_henze_exact` are the two test p-values; `k=3`.

Ranges below come from the CSV rows; the number in parentheses is the number of distinct checkpoints or seeds.

## Eval extraction (running BatchNorm statistics) · 10,000 points per split

| File | Checkpoints | Test seeds |
| --- | --- | --- |
| [knn_zoo_c100_m1_n10000_part_c0-39.csv](eval_mode/points_per_split_10000/knn_zoo_c100_m1_n10000_part_c0-39.csv) | 0–39 (40) | 42–91 (50) |
| [knn_zoo_c100_m1_n10000_part_c40-60.csv](eval_mode/points_per_split_10000/knn_zoo_c100_m1_n10000_part_c40-60.csv) | 40–60 (21) | 42–91 (50) |

## Eval extraction (running BatchNorm statistics) · 1,500 points per split

| File | Checkpoints | Test seeds |
| --- | --- | --- |
| [knn_zoo_c100_m1_n1500_part_c0-39.csv](eval_mode/points_per_split_1500/knn_zoo_c100_m1_n1500_part_c0-39.csv) | 0–39 (40) | 42–91 (50) |
| [knn_zoo_c100_m1_n1500_part_c40-60.csv](eval_mode/points_per_split_1500/knn_zoo_c100_m1_n1500_part_c40-60.csv) | 40–60 (21) | 42–91 (50) |

## Train extraction (current-batch BatchNorm statistics) · 10,000 points per split

| File | Checkpoints | Test seeds |
| --- | --- | --- |
| [knn_zoo_c100_m1_bntrain_n10000_part_c0-39.csv](train_mode/points_per_split_10000/knn_zoo_c100_m1_bntrain_n10000_part_c0-39.csv) | 0–39 (40) | 42–91 (50) |
| [knn_zoo_c100_m1_bntrain_n10000_part_c40-60.csv](train_mode/points_per_split_10000/knn_zoo_c100_m1_bntrain_n10000_part_c40-60.csv) | 40–60 (21) | 42–91 (50) |

## Train extraction (current-batch BatchNorm statistics) · 1,500 points per split

| File | Checkpoints | Test seeds |
| --- | --- | --- |
| [knn_zoo_c100_m1_bntrain_n1500_part_c0-39.csv](train_mode/points_per_split_1500/knn_zoo_c100_m1_bntrain_n1500_part_c0-39.csv) | 0–39 (40) | 42–91 (50) |
| [knn_zoo_c100_m1_bntrain_n1500_part_c40-60.csv](train_mode/points_per_split_1500/knn_zoo_c100_m1_bntrain_n1500_part_c40-60.csv) | 40–60 (21) | 42–91 (50) |
