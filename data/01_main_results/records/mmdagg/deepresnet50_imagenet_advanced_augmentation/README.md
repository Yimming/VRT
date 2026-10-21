# DeepResNet-50 / ImageNet: advanced augmentation

MMDAgg.

`p` records the MMDAgg rejection decision (0 or 1). It is not a p-value.

Ranges below come from the CSV rows; the number in parentheses is the number of distinct checkpoints or seeds.

## Eval extraction (running BatchNorm statistics) · 1,500 points per split

| File | Checkpoints | Test seeds |
| --- | --- | --- |
| [mmdagg_imgnet_adv_b1500_c1-300.csv](eval_mode/points_per_split_1500/mmdagg_imgnet_adv_b1500_c1-300.csv) | 1–300 (9) | 42–91 (50) |

## Eval extraction (running BatchNorm statistics) · 5,000 points per split

| File | Checkpoints | Test seeds |
| --- | --- | --- |
| [mmdagg_imgnet_adv_b5000_c1-5.csv](eval_mode/points_per_split_5000/mmdagg_imgnet_adv_b5000_c1-5.csv) | 1–5 (2) | 42–91 (50) |
| [mmdagg_imgnet_adv_b5000_c20-50.csv](eval_mode/points_per_split_5000/mmdagg_imgnet_adv_b5000_c20-50.csv) | 20–50 (2) | 42–91 (50) |
| [mmdagg_imgnet_adv_b5000_c100-150.csv](eval_mode/points_per_split_5000/mmdagg_imgnet_adv_b5000_c100-150.csv) | 100–150 (2) | 42–91 (50) |
| [mmdagg_imgnet_adv_b5000_c200-250.csv](eval_mode/points_per_split_5000/mmdagg_imgnet_adv_b5000_c200-250.csv) | 200–250 (2) | 42–91 (50) |
| [mmdagg_imgnet_adv_b5000_c300-300.csv](eval_mode/points_per_split_5000/mmdagg_imgnet_adv_b5000_c300-300.csv) | 300 | 42–91 (50) |

## Train extraction (current-batch BatchNorm statistics) · 1,500 points per split

| File | Checkpoints | Test seeds |
| --- | --- | --- |
| [mmdagg_imgnet_adv_bntrain_b1500_c1-300.csv](train_mode/points_per_split_1500/mmdagg_imgnet_adv_bntrain_b1500_c1-300.csv) | 1–300 (9) | 42–91 (50) |

## Train extraction (current-batch BatchNorm statistics) · 5,000 points per split

| File | Checkpoints | Test seeds |
| --- | --- | --- |
| [mmdagg_imgnet_adv_bntrain_b5000_c1.csv](train_mode/points_per_split_5000/mmdagg_imgnet_adv_bntrain_b5000_c1.csv) | 1 | 42–91 (50) |
| [mmdagg_imgnet_adv_bntrain_b5000_c5.csv](train_mode/points_per_split_5000/mmdagg_imgnet_adv_bntrain_b5000_c5.csv) | 5 | 42–91 (50) |
| [mmdagg_imgnet_adv_bntrain_b5000_c20.csv](train_mode/points_per_split_5000/mmdagg_imgnet_adv_bntrain_b5000_c20.csv) | 20 | 42–91 (50) |
| [mmdagg_imgnet_adv_bntrain_b5000_c50.csv](train_mode/points_per_split_5000/mmdagg_imgnet_adv_bntrain_b5000_c50.csv) | 50 | 42–91 (50) |
| [mmdagg_imgnet_adv_bntrain_b5000_c100.csv](train_mode/points_per_split_5000/mmdagg_imgnet_adv_bntrain_b5000_c100.csv) | 100 | 42–91 (50) |
| [mmdagg_imgnet_adv_bntrain_b5000_c150.csv](train_mode/points_per_split_5000/mmdagg_imgnet_adv_bntrain_b5000_c150.csv) | 150 | 42–91 (50) |
| [mmdagg_imgnet_adv_bntrain_b5000_c200.csv](train_mode/points_per_split_5000/mmdagg_imgnet_adv_bntrain_b5000_c200.csv) | 200 | 42–91 (50) |
| [mmdagg_imgnet_adv_bntrain_b5000_c250.csv](train_mode/points_per_split_5000/mmdagg_imgnet_adv_bntrain_b5000_c250.csv) | 250 | 42–91 (50) |
| [mmdagg_imgnet_adv_bntrain_b5000_c300.csv](train_mode/points_per_split_5000/mmdagg_imgnet_adv_bntrain_b5000_c300.csv) | 300 | 42–91 (50) |
