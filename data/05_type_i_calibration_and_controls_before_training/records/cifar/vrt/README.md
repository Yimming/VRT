# CIFAR random-network VRT controls

Choose a model, then the extraction mode and number of points per split.

| Model | Settings | Files |
| --- | --- | --- |
| [ResNet20-FRN / CIFAR-10](resnet20_frn_cifar10) | No BatchNorm; Official and random splits in each CSV | 5 |
| [ResNet-50 / CIFAR-100](resnet50_cifar100) | Eval extraction (running BatchNorm statistics); Official and random splits in each CSV | 5 |
| [SimCLR ResNet-50 / CIFAR-10](simclr_resnet50_cifar10) | Eval extraction (running BatchNorm statistics); Official and random splits in each CSV | 5 |
| [ResNet-18 / CIFAR-10](resnet18_cifar10) | Eval extraction (running BatchNorm statistics); Official and random splits in each CSV | 5 |
| [ResNet-18 / CIFAR-100](resnet18_cifar100) | Eval extraction (running BatchNorm statistics); Official and random splits in each CSV | 5 |

The CSV `p` column contains per-test p-values.
