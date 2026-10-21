# 5-layer CNN / CIFAR-10

Schilling and Henze.

`p_schilling` and `p_henze_exact` are the two test p-values; `k=3`.

Ranges below come from the CSV rows; the number in parentheses is the number of distinct checkpoints or seeds.

## No BatchNorm · 10,000 points per split

| File | Checkpoints | Test seeds |
| --- | --- | --- |
| [knn_wp_n10000_part_c0-390.csv](no_batchnorm/points_per_split_10000/knn_wp_n10000_part_c0-390.csv) | 0–390 (40) | 42–91 (50) |
| [knn_wp_n10000_part_c400-790.csv](no_batchnorm/points_per_split_10000/knn_wp_n10000_part_c400-790.csv) | 400–790 (40) | 42–91 (50) |
| [knn_wp_n10000_part_c800-1000.csv](no_batchnorm/points_per_split_10000/knn_wp_n10000_part_c800-1000.csv) | 800–1000 (21) | 42–91 (50) |

## No BatchNorm · 1,500 points per split

| File | Checkpoints | Test seeds |
| --- | --- | --- |
| [knn_wp_n1500_part_c0-390.csv](no_batchnorm/points_per_split_1500/knn_wp_n1500_part_c0-390.csv) | 0–390 (40) | 42–91 (50) |
| [knn_wp_n1500_part_c400-790.csv](no_batchnorm/points_per_split_1500/knn_wp_n1500_part_c400-790.csv) | 400–790 (40) | 42–91 (50) |
| [knn_wp_n1500_part_c800-1000.csv](no_batchnorm/points_per_split_1500/knn_wp_n1500_part_c800-1000.csv) | 800–1000 (21) | 42–91 (50) |
