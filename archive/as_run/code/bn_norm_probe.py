"""
bn_norm_probe.py — does BatchNorm alone move feature lengths? (nxR4-Q2 aid)
===========================================================================
Straightest possible design: SAME images (test set), SAME checkpoint
weights, only the BN mode differs.
  Pass A: model.eval()            -> BN uses stored running statistics
  Pass B: BN layers set to train  -> BN uses per-batch statistics
Everything else identical (loader order fixed, shuffle=False, no_grad).
Any change in feature norms between A and B is caused by BN's statistics
choice alone — zero change in data, zero change in weights.

Runs on EVERY model family. Models without BN (v7 uses FRN, wp has no norm
layer at all) are probed with the identical code path and must print
d_med% = +0.00 exactly — the built-in negative control.

Per (model, epoch, batch-size-B) we report:
  n_bn                          = number of BatchNorm modules found
  med_normA, med_normB, d_med%  = 100*(medB/medA - 1)
  MAD_A, MAD_B, d_disp%         = 100*(MADB/MADA - 1)
  cos_med, cos_p05              = per-sample cosine between A- and B-features
Pass B is run at two batch sizes (512 and 100) to show that even batch
COMPOSITION changes the lengths.

Run (GPU node, from the code dir so extract_feats.py imports):
  python bn_norm_probe.py --model r50 --epochs 30 61 120 200
  python bn_norm_probe.py --model all
Output: bn_norm_probe_{model}.csv in --outdir (default ladder_out).
"""
import argparse, csv, os
import numpy as np
import torch, torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

from extract_feats import (DEVICE, DATA, R50_DIR, V7_DIR, REG_ROOT,
                           ZOO_C10, ZOO_C100, C10_STATS, C100_STATS,
                           ResNet20FRN, WPCNN, parse_wp_config, find_wp_dir,
                           load_wp_ckpt, loaders)

DEFAULT_EPOCHS = {
    'r50': [30, 61, 120, 200], 'v7': [19, 100], 'wp': [500, 990],
    'ablA': [31], 'ablB': [30], 'ablC': [35], 'ablD': [30],
    'zoo_c10': [25], 'zoo_c100': [30],
}

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True,
                choices=['all'] + list(DEFAULT_EPOCHS.keys()))
ap.add_argument('--epochs', type=int, nargs='+', default=None)
ap.add_argument('--submodel', type=int, default=0, help='zoo model index')
ap.add_argument('--outdir', default='/scratch/p318770/model_zoos/ladder_out')
a = ap.parse_args()
print(f'DEVICE = {DEVICE}', flush=True)


def build(model_name):
    """Return (model, test_loader_dataset_name, load_ckpt_fn)."""
    if model_name == 'r50':
        m = torchvision.models.resnet50(weights=None, num_classes=100)
        m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False); m.maxpool = nn.Identity()
        def load(ep):
            ck = os.path.join(R50_DIR, f'checkpoint_{ep:03d}.pt')
            if not os.path.exists(ck): return None
            sd = torch.load(ck, map_location='cpu')
            return sd['model'] if isinstance(sd, dict) and 'model' in sd else sd
        return m, 'c100', load
    if model_name == 'v7':
        from safetensors.torch import load_file
        m = ResNet20FRN()
        def load(ep):
            ck = os.path.join(V7_DIR, f'version_7_epoch_{ep}.safetensors')
            return load_file(ck) if os.path.exists(ck) else None
        return m, 'c10', load
    if model_name == 'wp':
        wp_dir = find_wp_dir()
        m = WPCNN(parse_wp_config(os.path.join(wp_dir, 'params.json')))
        def load(ep):
            ck = os.path.join(wp_dir, f'checkpoint_{ep:06d}', 'checkpoint.pt')
            return load_wp_ckpt(ck)['model'] if os.path.exists(ck) else None
        return m, 'c10', load
    if model_name in ('ablA', 'ablB', 'ablC', 'ablD'):
        sub = {'ablA': 'A_baseline', 'ablB': 'B_strong_wd',
               'ablC': 'C_strong_aug', 'ablD': 'D_wd_plus_aug'}[model_name]
        m = torchvision.models.resnet18(weights=None, num_classes=10)
        m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False); m.maxpool = nn.Identity()
        def load(ep):
            ck = os.path.join(REG_ROOT, sub, f'checkpoint_{ep:03d}.pt')
            if not os.path.exists(ck): return None
            sd = torch.load(ck, map_location='cpu')
            return sd['model'] if isinstance(sd, dict) and 'model' in sd else sd
        return m, 'c10', load
    # zoos
    c10 = model_name == 'zoo_c10'
    zoo_dir = ZOO_C10 if c10 else ZOO_C100
    subdirs = sorted(d for d in os.listdir(zoo_dir)
                     if os.path.isdir(os.path.join(zoo_dir, d))
                     and not d.startswith('.') and d.startswith('NN_'))
    mpath = os.path.join(zoo_dir, subdirs[a.submodel])
    m = torchvision.models.resnet18(weights=None, num_classes=10 if c10 else 100)
    m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False); m.maxpool = nn.Identity()
    def load(ep):
        ck = os.path.join(mpath, f'checkpoint_{ep:06d}', 'checkpoints')
        return torch.load(ck, map_location='cpu') if os.path.exists(ck) else None
    return m, 'c10' if c10 else 'c100', load


@torch.no_grad()
def extract(model, loader):
    out = []
    for x, _ in loader:
        x = x.to(DEVICE)
        if hasattr(model, 'forward_features'):        # v7, wp
            out.append(model.forward_features(x).cpu())
        else:                                          # torchvision resnets
            x = model.conv1(x); x = model.bn1(x); x = model.relu(x); x = model.maxpool(x)
            x = model.layer1(x); x = model.layer2(x); x = model.layer3(x); x = model.layer4(x)
            out.append(model.avgpool(x).flatten(1).cpu())
    return torch.cat(out, 0).numpy().astype(np.float64)


def bn_train_mode(model):
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.train()


def mad(v):
    return float(np.median(np.abs(v - np.median(v))))


def probe(model_name):
    model, ds, load = build(model_name)
    model = model.to(DEVICE)
    n_bn = sum(isinstance(m, nn.modules.batchnorm._BatchNorm)
               for m in model.modules())
    stats = C10_STATS if ds == 'c10' else C100_STATS
    import torchvision.transforms as T
    tfm = T.Compose([T.ToTensor(), T.Normalize(*stats)])
    cls = torchvision.datasets.CIFAR10 if ds == 'c10' else torchvision.datasets.CIFAR100
    te = cls(DATA, False, download=False, transform=tfm)

    tag = model_name + (f'_m{a.submodel}' if model_name.startswith('zoo') else '')
    print(f'\n== {tag}  ({n_bn} BatchNorm modules'
          + (' — no BN: expect d_med exactly +0.00)' if n_bn == 0 else ')'),
          flush=True)
    print('epoch bsB  n_bn   medA       medB       d_med%   MADA     MADB'
          '     d_disp%  cos_med  cos_p05', flush=True)
    rows = []
    epochs = a.epochs or DEFAULT_EPOCHS[model_name]
    for ep in epochs:
        sd = load(ep)
        if sd is None:
            print(f'ep{ep}: checkpoint missing, skipped', flush=True); continue
        for bsB in (512, 100):
            model.load_state_dict(sd)   # fresh buffers before every pass pair
            model.eval()
            A = extract(model, DataLoader(te, batch_size=512, shuffle=False,
                                          num_workers=4, pin_memory=True))
            bn_train_mode(model)
            B = extract(model, DataLoader(te, batch_size=bsB, shuffle=False,
                                          num_workers=4, pin_memory=True))
            nA, nB = np.linalg.norm(A, axis=1), np.linalg.norm(B, axis=1)
            cos = (A * B).sum(1) / (nA * nB)
            r = dict(model=tag, epoch=ep, bsB=bsB, n_bn=n_bn,
                     medA=np.median(nA), medB=np.median(nB),
                     d_med=100 * (np.median(nB) / np.median(nA) - 1),
                     MADA=mad(nA), MADB=mad(nB),
                     d_disp=100 * (mad(nB) / mad(nA) - 1) if mad(nA) > 0 else 0.0,
                     cos_med=float(np.median(cos)),
                     cos_p05=float(np.quantile(cos, 0.05)))
            rows.append(r)
            print(f"{ep:5d} {bsB:4d} {n_bn:5d} {r['medA']:10.4f} {r['medB']:10.4f} "
                  f"{r['d_med']:+8.2f} {r['MADA']:8.4f} {r['MADB']:8.4f} "
                  f"{r['d_disp']:+8.2f} {r['cos_med']:8.5f} {r['cos_p05']:8.5f}",
                  flush=True)
    return rows


models = list(DEFAULT_EPOCHS.keys()) if a.model == 'all' else [a.model]
all_rows = []
for mn in models:
    try:
        all_rows += probe(mn)
    except Exception as e:
        print(f'{mn}: FAILED ({e})', flush=True)

os.makedirs(a.outdir, exist_ok=True)
suffix = 'all' if a.model == 'all' else a.model
out_csv = os.path.join(a.outdir, f'bn_norm_probe_{suffix}.csv')
with open(out_csv, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
    w.writeheader(); w.writerows(all_rows)
print('\nwrote', out_csv, flush=True)
