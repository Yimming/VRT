# ResNet-50 / CIFAR-100 Phase Zoo: training batch 8

MMD.

The CSV `p` column contains per-test p-values.

Ranges below come from the CSV rows; the number in parentheses is the number of distinct checkpoints or seeds.

## Eval extraction (running BatchNorm statistics) · 10,000 points per split

| File | Checkpoints | Test seeds |
| --- | --- | --- |
| [mmd_phase_bs8_n10000_c100-150.csv](eval_mode/points_per_split_10000/mmd_phase_bs8_n10000_c100-150.csv) | 100–150 (3) | 42–91 (50) |

## Eval extraction (running BatchNorm statistics) · 1,500 points per split

| File | Checkpoints | Test seeds |
| --- | --- | --- |
| [mmd_phase_bs8_n1500_c100-150.csv](eval_mode/points_per_split_1500/mmd_phase_bs8_n1500_c100-150.csv) | 100–150 (3) | 42–91 (50) |

## Train extraction (current-batch BatchNorm statistics) · 10,000 points per split

| File | Checkpoints | Test seeds |
| --- | --- | --- |
| [mmd_phase_bs8_bntrain_n10000_c100-150.csv](train_mode/points_per_split_10000/mmd_phase_bs8_bntrain_n10000_c100-150.csv) | 100–150 (3) | 42–91 (50) |

## Train extraction (current-batch BatchNorm statistics) · 1,500 points per split

| File | Checkpoints | Test seeds |
| --- | --- | --- |
| [mmd_phase_bs8_bntrain_n1500_c100-150.csv](train_mode/points_per_split_1500/mmd_phase_bs8_bntrain_n1500_c100-150.csv) | 100–150 (3) | 42–91 (50) |
