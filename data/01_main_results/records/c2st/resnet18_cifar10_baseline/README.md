# ResNet-18 / CIFAR-10: A, baseline

C2ST.

The CSV `p` column contains per-test p-values.

Ranges below come from the CSV rows; the number in parentheses is the number of distinct checkpoints or seeds.

## Eval extraction (running BatchNorm statistics) · Multiple budgets in each CSV

Points per split: 1,500, 10,000.

| File | Checkpoints | Test seeds |
| --- | --- | --- |
| [c2st_ablA_b1500-10000_c0-49.csv](eval_mode/multiple_budgets/c2st_ablA_b1500-10000_c0-49.csv) | 0–49 (50) | 42–91 (50) |

## Train extraction (current-batch BatchNorm statistics) · Multiple budgets in each CSV

Points per split: 1,500, 10,000.

| File | Checkpoints | Test seeds |
| --- | --- | --- |
| [c2st_ablA_bntrain_b1500-10000_c0-49.csv](train_mode/multiple_budgets/c2st_ablA_bntrain_b1500-10000_c0-49.csv) | 0–49 (50) | 42–91 (50) |
