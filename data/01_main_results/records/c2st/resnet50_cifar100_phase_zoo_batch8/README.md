# ResNet-50 / CIFAR-100 Phase Zoo: training batch 8

C2ST.

The CSV `p` column contains per-test p-values.

Ranges below come from the CSV rows; the number in parentheses is the number of distinct checkpoints or seeds.

## Eval extraction (running BatchNorm statistics) · Multiple budgets in each CSV

Points per split: 1,500, 10,000.

| File | Checkpoints | Test seeds |
| --- | --- | --- |
| [c2st_phase_bs8_b1500-10000_c100-150.csv](eval_mode/multiple_budgets/c2st_phase_bs8_b1500-10000_c100-150.csv) | 100–150 (3) | 42–91 (50) |

## Train extraction (current-batch BatchNorm statistics) · Multiple budgets in each CSV

Points per split: 1,500, 10,000.

| File | Checkpoints | Test seeds |
| --- | --- | --- |
| [c2st_phase_bs8_bntrain_b1500-10000_c100-150.csv](train_mode/multiple_budgets/c2st_phase_bs8_bntrain_b1500-10000_c100-150.csv) | 100–150 (3) | 42–91 (50) |
