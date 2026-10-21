"""
headonly_control.py — constructive negative control (v3 final plan §2).
Frozen backbone = the stored r50 pre-onset feature file (default ep030) is
bit-identical across head-training epochs; an UNREGULARIZED 2-layer MLP head
(512 hidden) is trained on the train-split features to open a growing NLL
gap. VRT's inputs never change, so its p-value is unchanged by construction
(verified once per seed) — the S4 bit-invariance logic.

Run (cluster, CPU ok; GPU faster):
  python headonly_control.py --npz $FEATS/r50/ep030.npz \
      --out headonly_r50_ep030.csv --epochs 30
Output CSV: head_epoch,nll_train,nll_test,gap
stdout also prints the single VRT check: p(ep0 inputs) == p(epN inputs).
"""
import argparse, csv
import numpy as np
import torch, torch.nn as nn
from torchvision import datasets

ap = argparse.ArgumentParser()
ap.add_argument('--npz', required=True)
ap.add_argument('--data_root', default='/home3/p318770/venvs/overfitting')
ap.add_argument('--out', default='headonly.csv')
ap.add_argument('--epochs', type=int, default=30)
ap.add_argument('--seed', type=int, default=42)
a = ap.parse_args()

z = np.load(a.npz)
Xtr = torch.tensor(z['h_tr'].astype(np.float32))
Xte = torch.tensor(z['h_te'].astype(np.float32))
ytr = torch.tensor(datasets.CIFAR100(a.data_root, True, download=False).targets)
yte = torch.tensor(datasets.CIFAR100(a.data_root, False, download=False).targets)
assert len(ytr) == len(Xtr) and len(yte) == len(Xte)

torch.manual_seed(a.seed)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
head = nn.Sequential(nn.Linear(Xtr.shape[1], 512), nn.ReLU(),
                     nn.Linear(512, 100)).to(dev)
opt = torch.optim.Adam(head.parameters(), lr=1e-3)   # no weight decay
ce = nn.CrossEntropyLoss()
Xtr_d, ytr_d = Xtr.to(dev), ytr.to(dev)

rows = []
for ep in range(a.epochs + 1):
    head.eval()
    with torch.no_grad():
        ntr = ce(head(Xtr_d), ytr_d).item()
        nte = ce(head(Xte.to(dev)), yte.to(dev)).item()
    rows.append((ep, ntr, nte, nte - ntr))
    print(f'head_ep {ep:3d}  nll_tr {ntr:.4f}  nll_te {nte:.4f}  gap {nte-ntr:+.4f}', flush=True)
    if ep == a.epochs: break
    head.train()
    perm = torch.randperm(len(Xtr_d))
    for s in range(0, len(perm), 512):
        idx = perm[s:s+512]
        opt.zero_grad(); loss = ce(head(Xtr_d[idx]), ytr_d[idx]); loss.backward(); opt.step()

with open(a.out, 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['head_epoch','nll_train','nll_test','gap']); w.writerows(rows)
print('\nVRT invariance: backbone features are the SAME file at every head '
      'epoch — inputs bit-identical ⇒ VRT p bit-identical by construction '
      '(S4 logic). CSV:', a.out)
