# MMD on the VRT score

Choose a model, then the extraction mode and number of points per split.

| Model | Settings | Files |
| --- | --- | --- |
| [ResNet20-FRN / CIFAR-10](resnet20_frn_cifar10) | No BatchNorm | 21 |
| [ResNet-18 / CIFAR-100 zoo: model 0](resnet18_cifar100_zoo_model0) | Eval extraction (running BatchNorm statistics) | 7 |
| [ResNet-18 / CIFAR-100 zoo: model 1](resnet18_cifar100_zoo_model1) | Eval extraction (running BatchNorm statistics) | 7 |
| [ResNet-18 / CIFAR-100 zoo: model 2](resnet18_cifar100_zoo_model2) | Eval extraction (running BatchNorm statistics) | 7 |
| [ResNet-18 / CIFAR-100 zoo: model 3](resnet18_cifar100_zoo_model3) | Eval extraction (running BatchNorm statistics) | 7 |
| [ResNet-18 / CIFAR-100 zoo: model 4](resnet18_cifar100_zoo_model4) | Eval extraction (running BatchNorm statistics) | 7 |
| [ResNet-18 / CIFAR-10 zoo: model 0](resnet18_cifar10_zoo_model0) | Eval extraction (running BatchNorm statistics) | 6 |
| [ResNet-18 / CIFAR-10 zoo: model 1](resnet18_cifar10_zoo_model1) | Eval extraction (running BatchNorm statistics) | 6 |
| [ResNet-18 / CIFAR-10 zoo: model 2](resnet18_cifar10_zoo_model2) | Eval extraction (running BatchNorm statistics) | 6 |
| [ResNet-18 / CIFAR-10 zoo: model 3](resnet18_cifar10_zoo_model3) | Eval extraction (running BatchNorm statistics) | 6 |
| [ResNet-18 / CIFAR-10 zoo: model 4](resnet18_cifar10_zoo_model4) | Eval extraction (running BatchNorm statistics) | 6 |
| [ResNet-50 / CIFAR-100](resnet50_cifar100) | Eval extraction (running BatchNorm statistics) | 21 |
| [ResNet-50 / CIFAR-100 Phase Zoo: training batch 8](resnet50_cifar100_phase_zoo_batch8) | Eval extraction (running BatchNorm statistics) | 1 |
| [ResNet-50 / CIFAR-100 Phase Zoo: training batch 128](resnet50_cifar100_phase_zoo_batch128) | Eval extraction (running BatchNorm statistics) | 1 |
| [ResNet-50 / CIFAR-100 Phase Zoo: training batch 1024](resnet50_cifar100_phase_zoo_batch1024) | Eval extraction (running BatchNorm statistics) | 1 |
| [SimCLR ResNet-50 / CIFAR-10](simclr_resnet50_cifar10) | Eval extraction (running BatchNorm statistics) | 53 |
| [SimCLR ResNet-50 / CIFAR-100](simclr_resnet50_cifar100) | Eval extraction (running BatchNorm statistics) | 53 |
| [ResNet-18 / CIFAR-10: A, baseline](resnet18_cifar10_baseline) | Eval extraction (running BatchNorm statistics) | 5 |
| [ResNet-18 / CIFAR-10: B, 10× weight decay](resnet18_cifar10_10x_weight_decay) | Eval extraction (running BatchNorm statistics) | 5 |
| [ResNet-18 / CIFAR-10: C, strong augmentation](resnet18_cifar10_strong_augmentation) | Eval extraction (running BatchNorm statistics) | 5 |
| [ResNet-18 / CIFAR-10: D, weight decay + augmentation](resnet18_cifar10_weight_decay_and_augmentation) | Eval extraction (running BatchNorm statistics) | 5 |
| [5-layer CNN / CIFAR-10](5layer_cnn_cifar10) | No BatchNorm | 10 |
| [DeepResNet-50 / ImageNet: simple augmentation](deepresnet50_imagenet_simple_augmentation) | Eval extraction (running BatchNorm statistics) | 2 |
| [DeepResNet-50 / ImageNet: advanced augmentation](deepresnet50_imagenet_advanced_augmentation) | Eval extraction (running BatchNorm statistics) | 2 |

The CSV `p` column contains per-test p-values.
