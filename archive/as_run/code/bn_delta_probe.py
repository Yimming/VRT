"""
bn_delta_probe.py - measure the per-layer BN mis-standardization directly
=========================================================================
(nxR4-Q2 backup measurement; existing checkpoints only, no training)

The protocol-switch runs verified the EFFECT (norm onsets move when the
BN statistics change). This probe measures the CAUSE, layer by layer.

Pure eval-mode forward (the deployed representation). A pre-hook on every
BatchNorm module accumulates exact per-channel mean/var of its INPUT over
a full split. Per channel at each layer:
  rho = sqrt((var_actual+eps)/(running_var+eps))  scale calibration, 1=calibrated
  m   = (mean_actual-running_mean)/sqrt(running_var+eps)  mean offset, std units
  tau = sqrt((var_test+eps)/(var_train+eps))  real test-vs-train activation
        scale gap at that depth; eval mode divides both splits by the SAME
        running sigma so this gap passes through to embedding lengths,
        train-mode extraction absorbs it at every layer
Reported per layer: channel medians. Detail -> CSV, stdout -> per-stage
summary. Anchor from the same pass: median embedding norms train/test
(r50 ep200 must land in 13.4-13.9 and ratio in 0.926-0.948).
"""
import argparse, csv, os
import numpy as np
import torch, torch.nn as nn
import torchvision

from extract_feats import (DEVICE, R50_DIR, REG_ROOT, C10_STATS, C100_STATS,
                           loaders)

BN = nn.modules.batchnorm._BatchNorm


class Tap:
    def __init__(self, name, mod):
        self.name = name
        C = mod.running_mean.numel()
        self.s = torch.zeros(C, dtype=torch.float64, device=DEVICE)
        self.q = torch.zeros(C, dtype=torch.float64, device=DEVICE)
        self.n = 0
        self.h = mod.register_forward_pre_hook(self.tap)

    def tap(self, mod, inp):
        x = inp[0].detach()
        if x.dim() == 4:
            self.s += x.sum((0, 2, 3), dtype=torch.float64)
            self.q += (x * x).sum((0, 2, 3), dtype=torch.float64)
            self.n += x.shape[0] * x.shape[2] * x.shape[3]
        else:
            self.s += x.sum(0, dtype=torch.float64)
            self.q += (x * x).sum(0, dtype=torch.float64)
            self.n += x.shape[0]

    def result(self):
        mu = (self.s / self.n).cpu().numpy()
        var = (self.q / self.n).cpu().numpy() - mu * mu
        return mu, np.clip(var, 0, None)

    def close(self):
        self.h.remove()


@torch.no_grad()
def sweep(model, loader):
    taps = [Tap(n, m) for n, m in model.named_modules() if isinstance(m, BN)]
    norms = []
    for x, _ in loader:
        x = x.to(DEVICE)
        x = model.conv1(x); x = model.bn1(x); x = model.relu(x); x = model.maxpool(x)
        x = model.layer1(x); x = model.layer2(x); x = model.layer3(x); x = model.layer4(x)
        f = model.avgpool(x).flatten(1)
        norms.append(f.norm(dim=1).cpu())
    out = {t.name: t.result() for t in taps}
    for t in taps:
        t.close()
    return out, torch.cat(norms).numpy()


def stage_of(name):
    for s in ('layer1', 'layer2', 'layer3', 'layer4'):
        if name.startswith(s):
            return s
    return 'stem'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='r50', choices=['r50', 'ablB', 'ablD'])
    ap.add_argument('--epochs', type=int, nargs='*', default=None)
    ap.add_argument('--outdir', default='/scratch/p318770/model_zoos/ladder_out')
    a = ap.parse_args()

    if a.model == 'r50':
        tr_l, te_l = loaders('c100', C100_STATS)
        model = torchvision.models.resnet50(weights=None, num_classes=100)
        ck_of = lambda ep: os.path.join(R50_DIR, f'checkpoint_{ep:03d}.pt')
        eps_default = [30, 61, 76, 120, 200]
    else:
        tr_l, te_l = loaders('c10', C10_STATS)
        model = torchvision.models.resnet18(weights=None, num_classes=10)
        sub = {'ablB': 'B_strong_wd', 'ablD': 'D_wd_plus_aug'}[a.model]
        ck_of = lambda ep: os.path.join(REG_ROOT, sub, f'checkpoint_{ep:03d}.pt')
        eps_default = [10, 25, 44, 49]
    model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    model.maxpool = nn.Identity()
    model = model.to(DEVICE)

    epochs = a.epochs if a.epochs else eps_default
    csv_path = os.path.join(a.outdir, f'bn_delta_{a.model}.csv')
    fh = open(csv_path, 'w', newline='')
    w = csv.writer(fh)
    w.writerow(['model', 'epoch', 'idx', 'layer', 'channels',
                'med_abs_rho1_tr', 'med_abs_rho1_te', 'med_rho_tr', 'med_rho_te',
                'med_tau', 'med_abs_m_tr', 'med_abs_m_te'])

    for ep in epochs:
        ck = ck_of(ep)
        if not os.path.exists(ck):
            print(f'MISSING {ck}', flush=True)
            continue
        sd = torch.load(ck, map_location='cpu')
        sd = sd['model'] if isinstance(sd, dict) and 'model' in sd else sd
        model.load_state_dict(sd)
        model.eval()

        tr_stats, tr_norms = sweep(model, tr_l)
        te_stats, te_norms = sweep(model, te_l)

        rows = []
        bnmods = [(n, m) for n, m in model.named_modules() if isinstance(m, BN)]
        for i, (name, mod) in enumerate(bnmods):
            rmu = mod.running_mean.detach().cpu().numpy()
            rvar = mod.running_var.detach().cpu().numpy()
            e = mod.eps
            mu_tr, v_tr = tr_stats[name]
            mu_te, v_te = te_stats[name]
            rho_tr = np.sqrt((v_tr + e) / (rvar + e))
            rho_te = np.sqrt((v_te + e) / (rvar + e))
            tau = np.sqrt((v_te + e) / (v_tr + e))
            m_tr = (mu_tr - rmu) / np.sqrt(rvar + e)
            m_te = (mu_te - rmu) / np.sqrt(rvar + e)
            row = dict(name=name,
                       ar1_tr=float(np.median(np.abs(rho_tr - 1))),
                       ar1_te=float(np.median(np.abs(rho_te - 1))),
                       rho_tr=float(np.median(rho_tr)),
                       rho_te=float(np.median(rho_te)),
                       tau=float(np.median(tau)),
                       am_tr=float(np.median(np.abs(m_tr))),
                       am_te=float(np.median(np.abs(m_te))))
            rows.append(row)
            w.writerow([a.model, ep, i, name, len(rmu),
                        f"{row['ar1_tr']:.5f}", f"{row['ar1_te']:.5f}",
                        f"{row['rho_tr']:.5f}", f"{row['rho_te']:.5f}",
                        f"{row['tau']:.5f}",
                        f"{row['am_tr']:.5f}", f"{row['am_te']:.5f}"])
        fh.flush()

        mtr = float(np.median(tr_norms)); mte = float(np.median(te_norms))
        print(f'\n== {a.model} ep{ep:03d}  n_bn={len(rows)}  '
              f'med_norm tr/te {mtr:.3f}/{mte:.3f}  ratio {mte/mtr:.4f}', flush=True)
        print('stage    layers  |rho-1|%tr  |rho-1|%te   tau-1%   |m|tr   |m|te')
        for s in ('stem', 'layer1', 'layer2', 'layer3', 'layer4'):
            rs = [r for r in rows if stage_of(r['name']) == s]
            if not rs:
                continue
            med = lambda k: float(np.median([r[k] for r in rs]))
            print(f'{s:8s} {len(rs):5d}  {100*med("ar1_tr"):9.2f}  '
                  f'{100*med("ar1_te"):9.2f}  {100*(med("tau")-1):+7.2f}  '
                  f'{med("am_tr"):.3f}   {med("am_te"):.3f}', flush=True)
        print('deepest: ' + ' | '.join(
            f'{r["name"]} tau{100*(r["tau"]-1):+.2f}% rho_te{100*(r["rho_te"]-1):+.2f}%'
            for r in rows[-3:]), flush=True)

    fh.close()
    print(f'\nCSV -> {csv_path}', flush=True)


main()
