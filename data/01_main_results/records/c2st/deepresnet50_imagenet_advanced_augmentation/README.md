# DeepResNet-50 / ImageNet: advanced augmentation

C2ST.

The CSV `p` column contains per-test p-values.

Ranges below come from the CSV rows; the number in parentheses is the number of distinct checkpoints or seeds.

## Eval extraction (running BatchNorm statistics) · Multiple budgets in each CSV

Points per split: 1,500, 10,000.

| File | Checkpoints | Test seeds |
| --- | --- | --- |
| [c2st_imgnet_adv_b1500-10000_c1-300.csv](eval_mode/multiple_budgets/c2st_imgnet_adv_b1500-10000_c1-300.csv) | 1–300 (9) | 42–91 (50) |

## Train extraction (current-batch BatchNorm statistics) · Multiple budgets in each CSV

Points per split: 1,500, 10,000.

| File | Checkpoints | Test seeds |
| --- | --- | --- |
| [c2st_imgnet_adv_bntrain_b1500-10000_c1-300.csv](train_mode/multiple_budgets/c2st_imgnet_adv_bntrain_b1500-10000_c1-300.csv) | 1–300 (9) | 42–91 (50) |
