"""extract_bntrain2.py — BN train-mode re-extraction, round 2:
ablA, ablC, zoo_c10 m0-m4, zoo_c100 m0-m4 (all BN models not yet done).
Same protocol as extract_bntrain.py; v7/wp have no BN (bit-identical)."""
import os
import numpy as np
import torch, torch.nn as nn
import torchvision
from extract_feats import (DEVICE, REG_ROOT, ZOO_C10, ZOO_C100,
                           C10_STATS, C100_STATS, OUT_ROOT, loaders)

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
        if done_already(out_dir, ep): continue
        ck = ckpt_of(ep)
        if not os.path.exists(ck): continue
        sd = torch.load(ck, map_location='cpu')
        sd = sd['model'] if isinstance(sd, dict) and 'model' in sd else sd
        model.load_state_dict(sd); bn_train(model)
        h_tr = extract(model, tr_l)
        model.load_state_dict(sd); bn_train(model)
        h_te = extract(model, te_l)
        save(out_dir, ep, h_tr, h_te)
        print(f'{tag}_bntrain ep{ep:03d} done', flush=True)

def r18(nc):
    m = torchvision.models.resnet18(weights=None, num_classes=nc)
    m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False); m.maxpool = nn.Identity()
    return m.to(DEVICE)

# ablA / ablC (CIFAR-10)
tr_l, te_l = loaders('c10', C10_STATS)
m = r18(10)
for tag, sub in (('ablA', 'A_baseline'), ('ablC', 'C_strong_aug')):
    run(tag, m, lambda ep, s=sub: os.path.join(REG_ROOT, s, f'checkpoint_{ep:03d}.pt'),
        tr_l, te_l, range(0, 50))

# zoos
for zname, zdir, nc, nep, dsl in (('zoo_c10', ZOO_C10, 10, 51, ('c10', C10_STATS)),
                                  ('zoo_c100', ZOO_C100, 100, 61, ('c100', C100_STATS))):
    tr_l, te_l = loaders(*dsl)
    m = r18(nc)
    subdirs = sorted(d for d in os.listdir(zdir)
                     if os.path.isdir(os.path.join(zdir, d))
                     and not d.startswith('.') and d.startswith('NN_'))
    for mi in range(5):
        mpath = os.path.join(zdir, subdirs[mi])
        run(f'{zname}_m{mi}', m,
            lambda ep, p=mpath: os.path.join(p, f'checkpoint_{ep:06d}', 'checkpoints'),
            tr_l, te_l, range(nep))
print('all done', flush=True)
