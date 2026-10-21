"""
extract_bntrain.py — re-extract features with BN in TRAIN mode (batch stats)
============================================================================
Purpose (user request): the norm-rung onsets (r50@61, ablB@44, ablD@48) are
computed on eval-mode features, where BN's running statistics were estimated
on TRAIN data — the test split is inherently "out of calibration". If that
asymmetry manufactures the norm signal, the onset is a BN artifact, not a
distribution difference. Remedy = App. H's: extract with BN in train mode
(per-batch statistics, applied identically to both splits), then re-run the
identical permutation protocol and compare onsets.

Models: the three BN models with a norm onset (r50, ablB, ablD).
v7/wp have no BN -> train-mode extraction is bit-identical by construction.

Protocol: batch 512, shuffle=False, no_grad; state dict reloaded before each
split so running-buffer updates during a sweep cannot leak across splits.
Output: OUT_ROOT/{model}_bntrain/ep{N:03d}.npz  (same format as extract_feats)
"""
import os
import numpy as np
import torch, torch.nn as nn
import torchvision
from extract_feats import (DEVICE, R50_DIR, REG_ROOT, C10_STATS, C100_STATS,
                           OUT_ROOT, loaders)


def done_already(out_dir, ep):
    return os.path.exists(os.path.join(out_dir, f'ep{ep:03d}.npz'))

def save(out_dir, ep, h_tr, h_te):
    os.makedirs(out_dir, exist_ok=True)
    tmp = os.path.join(out_dir, f'.tmp_ep{ep:03d}.npz')
    np.savez(tmp, h_tr=h_tr, h_te=h_te)
    os.replace(tmp, os.path.join(out_dir, f'ep{ep:03d}.npz'))

print(f'DEVICE = {DEVICE}', flush=True)


def bn_train(model):
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.train()


@torch.no_grad()
def extract(model, loader):
    out = []
    for x, _ in loader:
        x = x.to(DEVICE)
        x = model.conv1(x); x = model.bn1(x); x = model.relu(x); x = model.maxpool(x)
        x = model.layer1(x); x = model.layer2(x); x = model.layer3(x); x = model.layer4(x)
        out.append(model.avgpool(x).flatten(1).cpu())
    return torch.cat(out, 0).numpy().astype(np.float32)


def run(tag, model, ckpt_of, tr_l, te_l, ep_range):
    out_dir = os.path.join(OUT_ROOT, f'{tag}_bntrain')
    for ep in ep_range:
        if done_already(out_dir, ep):
            continue
        ck = ckpt_of(ep)
        if not os.path.exists(ck):
            continue
        sd = torch.load(ck, map_location='cpu')
        sd = sd['model'] if isinstance(sd, dict) and 'model' in sd else sd
        model.load_state_dict(sd); bn_train(model)
        h_tr = extract(model, tr_l)
        model.load_state_dict(sd); bn_train(model)   # fresh buffers per split
        h_te = extract(model, te_l)
        save(out_dir, ep, h_tr, h_te)
        print(f'{tag}_bntrain ep{ep:03d} done', flush=True)


# r50 (CIFAR-100)
tr_l, te_l = loaders('c100', C100_STATS)
m = torchvision.models.resnet50(weights=None, num_classes=100)
m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False); m.maxpool = nn.Identity()
m = m.to(DEVICE)
run('r50', m, lambda ep: os.path.join(R50_DIR, f'checkpoint_{ep:03d}.pt'),
    tr_l, te_l, range(0, 201))

# ablB / ablD (CIFAR-10)
tr_l, te_l = loaders('c10', C10_STATS)
m = torchvision.models.resnet18(weights=None, num_classes=10)
m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False); m.maxpool = nn.Identity()
m = m.to(DEVICE)
for tag, sub in (('ablB', 'B_strong_wd'), ('ablD', 'D_wd_plus_aug')):
    run(tag, m,
        lambda ep, s=sub: os.path.join(REG_ROOT, s, f'checkpoint_{ep:03d}.pt'),
        tr_l, te_l, range(0, 50))

print('all done', flush=True)
