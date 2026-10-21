# ResNet-18 / CIFAR-100 zoo: model 0

C2ST.

The CSV `p` column contains per-test p-values.

Ranges below come from the CSV rows; the number in parentheses is the number of distinct checkpoints or seeds.

## Eval extraction (running BatchNorm statistics) · Multiple budgets in each CSV

Points per split: 1,500, 10,000.

| File | Checkpoints | Test seeds |
| --- | --- | --- |
| [c2st_zoo_c100_m0_b1500-10000_c0-60.csv](eval_mode/multiple_budgets/c2st_zoo_c100_m0_b1500-10000_c0-60.csv) | 0–60 (61) | 42–91 (50) |

## Train extraction (current-batch BatchNorm statistics) · Multiple budgets in each CSV

Points per split: 1,500, 10,000.

| File | Checkpoints | Test seeds |
| --- | --- | --- |
| [c2st_zoo_c100_m0_bntrain_b1500-10000_c0-60.csv](train_mode/multiple_budgets/c2st_zoo_c100_m0_bntrain_b1500-10000_c0-60.csv) | 0–60 (61) | 42–91 (50) |
