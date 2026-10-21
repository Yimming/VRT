"""
bntrain_extract.py — SimCLR encoder features with BatchNorm in train mode (per-batch statistics), for the eval-mode versus
train-mode BatchNorm table. Same protocol as the CIFAR train-mode features (extract_bntrain.py, extract_bntrain2.py):
the network is put in eval mode and only its BatchNorm layers in train mode, the checkpoint is reloaded before each split
so that no running statistic carries over from one split to the other, and each split is passed in its stored order in
batches of 512 (no shuffling), so every batch holds images of one split only. Checkpoint loading, preprocessing and the
features used (h = model.encoder(x), 2,048-D) are those of supcon_fill_vrt.extract, which gives the eval-mode features.
Imported by ssl_s50_fill.py and ssl_mmdagg5k_seeds.py when they run with --bn_train.
"""
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as T
from torch.utils.data import DataLoader

from supcon_fill_vrt import MEAN_STD, DATA_FOLDER, load_ckpt

BATCH = 512


def bn_train(model):
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.train()


@torch.no_grad()
def extract_bntrain(model, ckpt_path, dataset, device):
    mean, std = MEAN_STD[dataset]
    tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    cls = torchvision.datasets.CIFAR10 if dataset == "cifar10" else torchvision.datasets.CIFAR100
    out = {}
    for split, train in (("tr", True), ("te", False)):
        load_ckpt(model, ckpt_path, device)          # fresh running statistics for each split
        bn_train(model)
        ds = cls(root=DATA_FOLDER, train=train, transform=tf, download=False)
        loader = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=8, pin_memory=True)
        hs = []
        for x, _ in loader:
            hs.append(model.encoder(x.to(device, non_blocking=True)).float().cpu())
        out[f"h_{split}"] = torch.cat(hs).numpy()
    return out
