# ResNet-18 / CIFAR-10 zoo: model 2

C2ST.

The CSV `p` column contains per-test p-values.

Ranges below come from the CSV rows; the number in parentheses is the number of distinct checkpoints or seeds.

## Eval extraction (running BatchNorm statistics) · Multiple budgets in each CSV

Points per split: 1,500, 10,000.

| File | Checkpoints | Test seeds |
| --- | --- | --- |
| [c2st_zoo_c10_m2_b1500-10000_c0-50.csv](eval_mode/multiple_budgets/c2st_zoo_c10_m2_b1500-10000_c0-50.csv) | 0–50 (51) | 42–91 (50) |

## Train extraction (current-batch BatchNorm statistics) · Multiple budgets in each CSV

Points per split: 1,500, 10,000.

| File | Checkpoints | Test seeds |
| --- | --- | --- |
| [c2st_zoo_c10_m2_bntrain_b1500-10000_c0-50.csv](train_mode/multiple_budgets/c2st_zoo_c10_m2_bntrain_b1500-10000_c0-50.csv) | 0–50 (51) | 42–91 (50) |
