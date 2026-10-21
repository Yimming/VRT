# DeepResNet-50 / ImageNet: advanced augmentation

VRT.

The CSV `p` column contains per-test p-values.

Ranges below come from the CSV rows; the number in parentheses is the number of distinct checkpoints or seeds.

## Eval extraction (running BatchNorm statistics) · 10,000 points per split

| File | Checkpoints | Test seeds |
| --- | --- | --- |
| [vrt_fwd_imgnet_adv_large_c1-300.csv](eval_mode/points_per_split_10000/vrt_fwd_imgnet_adv_large_c1-300.csv) | 1–300 (9) | 42–91 (50) |

## Eval extraction (running BatchNorm statistics) · 1,500 points per split

| File | Checkpoints | Test seeds |
| --- | --- | --- |
| [vrt_fwd_imgnet_adv_small_c1-300.csv](eval_mode/points_per_split_1500/vrt_fwd_imgnet_adv_small_c1-300.csv) | 1–300 (9) | 42–91 (50) |

## Train extraction (current-batch BatchNorm statistics) · 10,000 points per split

| File | Checkpoints | Test seeds |
| --- | --- | --- |
| [vrt_fwd_imgnet_adv_bntrain_large_c1-300.csv](train_mode/points_per_split_10000/vrt_fwd_imgnet_adv_bntrain_large_c1-300.csv) | 1–300 (9) | 42–91 (50) |

## Train extraction (current-batch BatchNorm statistics) · 1,500 points per split

| File | Checkpoints | Test seeds |
| --- | --- | --- |
| [vrt_fwd_imgnet_adv_bntrain_small_c1-300.csv](train_mode/points_per_split_1500/vrt_fwd_imgnet_adv_bntrain_small_c1-300.csv) | 1–300 (9) | 42–91 (50) |
