# SimCLR: train-mode extraction

Choose a model, then the extraction mode and number of points per split.

| Model | Settings | Files |
| --- | --- | --- |
| [SimCLR ResNet-50 / CIFAR-10](simclr_resnet50_cifar10) | 1,500 points per split; 10,000 points per split; 5,000 points per split; MMD and ED on raw features; MMD, ED and C2ST; MMDAgg; Multiple budgets in each CSV; Schilling and Henze; VRT; VRT, MMD, ED and C2ST | 344 |
| [SimCLR ResNet-50 / CIFAR-100](simclr_resnet50_cifar100) | 1,500 points per split; 10,000 points per split; 5,000 points per split; MMD and ED on raw features; MMD, ED and C2ST; MMDAgg; Multiple budgets in each CSV; Schilling and Henze; VRT | 418 |

`p` is a p-value except for MMDAgg, where it records a 0/1 rejection decision.
