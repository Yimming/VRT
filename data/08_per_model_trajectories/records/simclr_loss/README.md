# SimCLR loss curves

Choose a model, then a checkpoint range. Each CSV contains both BatchNorm extraction modes.

| Model | Settings | Files |
| --- | --- | --- |
| [SimCLR ResNet-50 / CIFAR-10](simclr_resnet50_cifar10) | See CSV columns | 10 |
| [SimCLR ResNet-50 / CIFAR-100](simclr_resnet50_cifar100) | See CSV columns | 10 |

`tr_*`, `te_*` and `gap_*` contain training loss, test loss and test-minus-training loss. Figure 6 (Appendix F) uses `tr_eval` and `te_eval`, multiplied by `0.07 / 0.5 = 0.14` to convert the stored SupConLoss units to NT-Xent nats.
