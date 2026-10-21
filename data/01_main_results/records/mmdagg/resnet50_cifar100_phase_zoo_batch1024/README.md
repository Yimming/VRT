# ResNet-50 / CIFAR-100 Phase Zoo: training batch 1024

MMDAgg.

`p` records the MMDAgg rejection decision (0 or 1). It is not a p-value.

Ranges below come from the CSV rows; the number in parentheses is the number of distinct checkpoints or seeds.

## Eval extraction (running BatchNorm statistics) · 1,500 points per split

| File | Checkpoints | Test seeds |
| --- | --- | --- |
| [mmdagg_phase_bs1024_b1500_c100-150.csv](eval_mode/points_per_split_1500/mmdagg_phase_bs1024_b1500_c100-150.csv) | 100–150 (3) | 42–91 (50) |

## Eval extraction (running BatchNorm statistics) · 5,000 points per split

| File | Checkpoints | Test seeds |
| --- | --- | --- |
| [mmdagg_phase_bs1024_b5000_c100-125.csv](eval_mode/points_per_split_5000/mmdagg_phase_bs1024_b5000_c100-125.csv) | 100–125 (2) | 42–91 (50) |
| [mmdagg_phase_bs1024_b5000_c150-150.csv](eval_mode/points_per_split_5000/mmdagg_phase_bs1024_b5000_c150-150.csv) | 150 | 42–91 (50) |

## Train extraction (current-batch BatchNorm statistics) · 1,500 points per split

| File | Checkpoints | Test seeds |
| --- | --- | --- |
| [mmdagg_phase_bs1024_bntrain_b1500_c100-150.csv](train_mode/points_per_split_1500/mmdagg_phase_bs1024_bntrain_b1500_c100-150.csv) | 100–150 (3) | 42–91 (50) |

## Train extraction (current-batch BatchNorm statistics) · 5,000 points per split

| File | Checkpoints | Test seeds |
| --- | --- | --- |
| [mmdagg_phase_bs1024_bntrain_b5000_c100-100.csv](train_mode/points_per_split_5000/mmdagg_phase_bs1024_bntrain_b5000_c100-100.csv) | 100 | 42–91 (50) |
| [mmdagg_phase_bs1024_bntrain_b5000_c125-125.csv](train_mode/points_per_split_5000/mmdagg_phase_bs1024_bntrain_b5000_c125-125.csv) | 125 | 42–91 (50) |
| [mmdagg_phase_bs1024_bntrain_b5000_c150-150.csv](train_mode/points_per_split_5000/mmdagg_phase_bs1024_bntrain_b5000_c150-150.csv) | 150 | 42–91 (50) |
