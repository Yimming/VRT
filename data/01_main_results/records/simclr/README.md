# SimCLR raw test results

Some CSVs contain multiple tests or both budgets. Their file tables list the contents explicitly.

| Model | Eval extraction | Train extraction | MMDAgg, 5,000 points (eval) | Earlier runs |
| --- | --- | --- | --- | --- |
| SimCLR ResNet-50 / CIFAR-10 | [Open](eval/simclr_resnet50_cifar10) | [Open](train/simclr_resnet50_cifar10) | [Open](aggregation_and_earlier_runs/simclr_resnet50_cifar10) | [Legacy](aggregation_and_earlier_runs/legacy_runs/simclr_resnet50_cifar10) |
| SimCLR ResNet-50 / CIFAR-100 | [Open](eval/simclr_resnet50_cifar100) | [Open](train/simclr_resnet50_cifar100) | [Open](aggregation_and_earlier_runs/simclr_resnet50_cifar100) | [Legacy](aggregation_and_earlier_runs/legacy_runs/simclr_resnet50_cifar100) |

Eval extraction uses saved BatchNorm running statistics; train extraction uses current-batch statistics. The checkpoint weights are unchanged.
