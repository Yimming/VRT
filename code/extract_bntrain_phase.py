"""
extract_bntrain_phase.py — Phase Zoo (ResNet-50 on CIFAR-100, training batch sizes 8, 128 and 1,024; checkpoints 100, 125
and 150) features with BatchNorm in train mode, for the eval-mode versus train-mode BatchNorm table. Same protocol as
extract_bntrain.py and extract_bntrain2.py: BatchNorm layers in train mode (rest of the network in eval mode), the
checkpoint reloaded before each split, each split in its stored order in batches of 512 (extract_feats.loaders). Model
construction and checkpoint loading as the eval-mode Phase Zoo extraction in extract_feats.py (--model phase).
Output: /scratch/p318770/model_zoos/ladder_feats/phase_<bs>_bntrain/ep<NNN>.npz with h_tr, h_te (float32, raw).
Usage (GPU node, from the code folder):  python extract_bntrain_phase.py
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
from extract_feats import DEVICE, PHASE_ZOO, PHASE_MODELS, C100_STATS, OUT_ROOT, loaders


def bn_train(model):
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.train()


@torch.no_grad()
def extract(model, loader):
    """penultimate features of the torchvision ResNet-50 (as extract_feats.extract_torchvision_resnet, without its
    model.eval() call, which would undo the train-mode BatchNorm)"""
    out = []
    for x, _ in loader:
        x = x.to(DEVICE)
        x = model.conv1(x); x = model.bn1(x); x = model.relu(x); x = model.maxpool(x)
        x = model.layer1(x); x = model.layer2(x); x = model.layer3(x); x = model.layer4(x)
        out.append(model.avgpool(x).flatten(1).cpu())
    return torch.cat(out, 0).numpy().astype(np.float32)


def save(out_dir, ep, h_tr, h_te):
    os.makedirs(out_dir, exist_ok=True)
    tmp = os.path.join(out_dir, f'.tmp_ep{ep:03d}.npz')
    np.savez(tmp, h_tr=h_tr, h_te=h_te)
    os.replace(tmp, os.path.join(out_dir, f'ep{ep:03d}.npz'))


def main():
    print(f'DEVICE = {DEVICE}', flush=True)
    tr_l, te_l = loaders('c100', C100_STATS)
    model = torchvision.models.resnet50(weights=None, num_classes=100)
    model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    model.maxpool = nn.Identity()
    model = model.to(DEVICE)
    for label, mname in PHASE_MODELS.items():
        out_dir = os.path.join(OUT_ROOT, f'phase_{label}_bntrain')
        for ep in (100, 125, 150):
            if os.path.exists(os.path.join(out_dir, f'ep{ep:03d}.npz')):
                print(f'phase_{label}_bntrain ep{ep:03d} exists, skip', flush=True); continue
            ck = os.path.join(PHASE_ZOO, mname, f'checkpoint_{ep:06d}', 'checkpoints')
            if not os.path.exists(ck):
                print(f'phase_{label} ep{ep}: missing checkpoint {ck}', flush=True); continue
            sd = torch.load(ck, map_location='cpu')
            sd = {k.replace('module.', ''): v for k, v in sd.items()}
            model.load_state_dict(sd); bn_train(model)
            h_tr = extract(model, tr_l)
            model.load_state_dict(sd); bn_train(model)
            h_te = extract(model, te_l)
            save(out_dir, ep, h_tr, h_te)
            print(f'phase_{label}_bntrain ep{ep:03d} done {h_tr.shape} {h_te.shape}', flush=True)


if __name__ == '__main__':
    main()
