"""
train_regularization_seed.py — Configurations A-D, seeded rerun
===============================================================
Identical recipe to the supplementary train_regularization.py (ResNet-18
CIFAR-10, SGD max lr 0.1, OneCycleLR, batch 128, 50 epochs, checkpoint
every epoch, 2x2 grid of weight decay {5e-4, 5e-3} x augmentation
{basic, AutoAugment+RandomErasing}), with a real --seed and a separate
save root per seed, for the multi-seed robustness check of the
interventional grid (Figure 3).

Usage:
  python train_regularization_seed.py --seed 1
  python train_regularization_seed.py --seed 2 --configs A_baseline,C_strong_aug
"""
import argparse
import csv
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

ap = argparse.ArgumentParser()
ap.add_argument('--seed', type=int, required=True)
ap.add_argument('--configs', default='A_baseline,B_strong_wd,C_strong_aug,D_wd_plus_aug')
args = ap.parse_args()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA = '/home3/p318770/venvs/overfitting'
SAVE_ROOT = f'/scratch/p318770/model_zoos/ablation_wd_aug_seed{args.seed}'
os.makedirs(SAVE_ROOT, exist_ok=True)

# ── Seeding (the only functional addition) ──
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

configs = {
    'A_baseline':     {'wd': 0.0005, 'aug': False},
    'B_strong_wd':    {'wd': 0.005,  'aug': False},
    'C_strong_aug':   {'wd': 0.0005, 'aug': True},
    'D_wd_plus_aug':  {'wd': 0.005,  'aug': True},
}
configs = {k: v for k, v in configs.items() if k in args.configs.split(',')}

norm = T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
tfm_basic = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                       T.ToTensor(), norm])
tfm_strong = T.Compose([
    T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
    T.AutoAugment(T.AutoAugmentPolicy.CIFAR10),
    T.ToTensor(), norm,
    T.RandomErasing(p=0.25),
])
tfm_test = T.Compose([T.ToTensor(), norm])

te_loader = DataLoader(
    torchvision.datasets.CIFAR10(DATA, False, download=False, transform=tfm_test),
    batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

for name, cfg in configs.items():
    print('=' * 60)
    print(f'Training: {name} (wd={cfg["wd"]}, aug={cfg["aug"]}, seed={args.seed})')
    print('=' * 60)
    save_dir = os.path.join(SAVE_ROOT, name)
    os.makedirs(save_dir, exist_ok=True)
    if os.path.exists(os.path.join(save_dir, 'checkpoint_049.pt')):
        print('  Already done, skip.'); continue
    tfm_tr = tfm_strong if cfg['aug'] else tfm_basic
    tr_loader = DataLoader(
        torchvision.datasets.CIFAR10(DATA, True, download=False, transform=tfm_tr),
        batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    model = torchvision.models.resnet18(weights=None, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model = model.to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                          weight_decay=cfg['wd'])
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, epochs=50,
                                              steps_per_epoch=len(tr_loader))
    criterion = nn.CrossEntropyLoss()
    json.dump({**cfg, 'seed': args.seed},
              open(os.path.join(save_dir, 'config.json'), 'w'))
    log = []
    for ep in range(50):
        t0 = time.time()
        model.train()
        tr_loss, tr_correct, tr_total = 0, 0, 0
        for x, y in tr_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            tr_loss += loss.item() * len(y)
            tr_correct += (out.argmax(1) == y).sum().item()
            tr_total += len(y)
        model.eval()
        te_correct, te_total, te_loss = 0, 0, 0
        with torch.no_grad():
            for x, y in te_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                te_loss += criterion(out, y).item() * len(y)
                te_correct += (out.argmax(1) == y).sum().item()
                te_total += len(y)
        tr_acc = tr_correct / tr_total
        te_acc = te_correct / te_total
        dt = time.time() - t0
        log.append({'ep': ep, 'tr_loss': tr_loss / tr_total,
                    'te_loss': te_loss / te_total,
                    'tr_acc': tr_acc, 'te_acc': te_acc, 'gap': tr_acc - te_acc})
        torch.save(model.state_dict(),
                   os.path.join(save_dir, f'checkpoint_{ep:03d}.pt'))
        print(f'  ep{ep:3d}: tr_acc={tr_acc:.4f} te_acc={te_acc:.4f} '
              f'gap={tr_acc-te_acc:+.4f} ({dt:.0f}s)', flush=True)
    with open(os.path.join(save_dir, 'progress.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=log[0].keys())
        w.writeheader(); w.writerows(log)
    print(f'  Saved to {save_dir}\n')

print('\nAll training done.')
