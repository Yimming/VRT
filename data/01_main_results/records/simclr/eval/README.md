# SimCLR: eval-mode extraction

Choose a model, then the extraction mode and number of points per split.

| Model | Settings | Files |
| --- | --- | --- |
| [SimCLR ResNet-50 / CIFAR-10](simclr_resnet50_cifar10) | 1,500 points per split; 10,000 points per split; MMD and ED on raw features; MMDAgg; Multiple budgets in each CSV; Reversed VRT, Schilling and Henze; VRT, MMD, ED and C2ST | 231 |
| [SimCLR ResNet-50 / CIFAR-100](simclr_resnet50_cifar100) | 1,500 points per split; 10,000 points per split; MMD and ED on raw features; MMDAgg; Multiple budgets in each CSV; Reversed VRT, Schilling and Henze; VRT, MMD, ED and C2ST | 233 |

`p` is a p-value except for MMDAgg, where it records a 0/1 rejection decision.
