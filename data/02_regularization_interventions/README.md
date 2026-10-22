# 4.3 Regularization interventions

Four ResNet-18/CIFAR-10 configurations A–D. [Summary CSV](regularization_interventions.csv).

## Per-seed results

These records are shared with Section 4.2 and stored in `01_main_results/records/`. Each link below leads to the CSVs for that configuration and method, grouped by extraction mode and sample size.

| Configuration | VRT | MMD | ED | C2ST | MMDAgg |
| --- | --- | --- | --- | --- | --- |
| A: baseline | [VRT](../01_main_results/records/vrt/resnet18_cifar10_baseline/) | [MMD](../01_main_results/records/mmd/resnet18_cifar10_baseline/) | [ED](../01_main_results/records/energy_distance/resnet18_cifar10_baseline/) | [C2ST](../01_main_results/records/c2st/resnet18_cifar10_baseline/) | [MMDAgg](../01_main_results/records/mmdagg/resnet18_cifar10_baseline/) |
| B: 10× weight decay | [VRT](../01_main_results/records/vrt/resnet18_cifar10_10x_weight_decay/) | [MMD](../01_main_results/records/mmd/resnet18_cifar10_10x_weight_decay/) | [ED](../01_main_results/records/energy_distance/resnet18_cifar10_10x_weight_decay/) | [C2ST](../01_main_results/records/c2st/resnet18_cifar10_10x_weight_decay/) | [MMDAgg](../01_main_results/records/mmdagg/resnet18_cifar10_10x_weight_decay/) |
| C: strong augmentation | [VRT](../01_main_results/records/vrt/resnet18_cifar10_strong_augmentation/) | [MMD](../01_main_results/records/mmd/resnet18_cifar10_strong_augmentation/) | [ED](../01_main_results/records/energy_distance/resnet18_cifar10_strong_augmentation/) | [C2ST](../01_main_results/records/c2st/resnet18_cifar10_strong_augmentation/) | [MMDAgg](../01_main_results/records/mmdagg/resnet18_cifar10_strong_augmentation/) |
| D: weight decay + augmentation | [VRT](../01_main_results/records/vrt/resnet18_cifar10_weight_decay_and_augmentation/) | [MMD](../01_main_results/records/mmd/resnet18_cifar10_weight_decay_and_augmentation/) | [ED](../01_main_results/records/energy_distance/resnet18_cifar10_weight_decay_and_augmentation/) | [C2ST](../01_main_results/records/c2st/resnet18_cifar10_weight_decay_and_augmentation/) | [MMDAgg](../01_main_results/records/mmdagg/resnet18_cifar10_weight_decay_and_augmentation/) |

The test records contain per-seed p-values; MMDAgg records binary rejection decisions instead. Training/test loss curves: [A](../../figures/benchmark_final/losses_ablation_A_baseline.csv), [B](../../figures/benchmark_final/losses_ablation_B_strong_wd.csv), [C](../../figures/benchmark_final/losses_ablation_C_strong_aug.csv), [D](../../figures/benchmark_final/losses_ablation_D_wd_plus_aug.csv).

Runner: `train.py regularization; evaluate.py test`.

[Filename and column guide](../README.md#names-used-in-the-files) explains `bntrain`, `raw`, budgets, checkpoints and seeds.
