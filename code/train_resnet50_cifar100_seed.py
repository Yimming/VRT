"""
train_resnet50_cifar100_seed.py — Train ResNet50 on CIFAR-100 (seeded rerun)
============================================================================
Identical recipe to the paper's train_resnet50_cifar100.py (SGD lr=0.1,
MultiStepLR [60,120,160] gamma=0.2, 200 epochs, batch 128, wd 5e-4,
checkpoint every epoch), with two additions for the rebuttal multi-seed
reproduction (reviewer Ftg8: "headline relies on a single self-trained
model"):

  --seed N : real random seed (init, shuffling, augmentation)
  saves to SAVE_ROOT/resnet50_cifar100_seed{N}/   (never touches the
  original run's directory)

Usage:
  python train_resnet50_cifar100_seed.py --seed 1
  python train_resnet50_cifar100_seed.py --seed 2
"""
import os
import time
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

ap = argparse.ArgumentParser()
ap.add_argument('--seed', type=int, required=True)
args = ap.parse_args()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA = '/home3/p318770/venvs/overfitting'
SAVE_DIR = f'/scratch/p318770/model_zoos/resnet50_cifar100_seed{args.seed}'
os.makedirs(SAVE_DIR, exist_ok=True)

# ── Seeding (the only functional addition) ──
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

# ── Hyperparameters (identical to original) ──
EPOCHS = 200
BATCH = 128
LR = 0.1
MOMENTUM = 0.9
WD = 5e-4
MILESTONES = [60, 120, 160]
GAMMA = 0.2  # lr / 5

# ── Data (identical) ──
train_tfm = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
])
test_tfm = T.Compose([
    T.ToTensor(),
    T.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
])
train_ds = torchvision.datasets.CIFAR100(DATA, True, download=True, transform=train_tfm)
test_ds = torchvision.datasets.CIFAR100(DATA, False, download=True, transform=test_tfm)
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

# ── Model (identical): ResNet50 for CIFAR ──
model = torchvision.models.resnet50(weights=None, num_classes=100)
model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
model.maxpool = nn.Identity()
model = model.to(DEVICE)
print(f"Model: ResNet50 for CIFAR-100, D=2048, seed={args.seed}")
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
print(f"Device: {DEVICE}")
print(f"Save: {SAVE_DIR}")
print(f"Config: epochs={EPOCHS}, batch={BATCH}, lr={LR}, wd={WD}")
print(f"Schedule: MultiStepLR milestones={MILESTONES}, gamma={GAMMA}")
print(f"Train: {len(train_ds)}, Test: {len(test_ds)}\n")

# ── Optimizer & Scheduler (identical) ──
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WD)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES, gamma=GAMMA)
criterion = nn.CrossEntropyLoss()

# ── Resume ──
# Release fix. checkpoint_N holds the model after N completed epochs, so training resumes with loop index N (the
# as-run version used N + 1, which trained epoch N + 2 next and never wrote checkpoint_{N+1}). checkpoint_000 is now
# written in the same dictionary format as every later checkpoint (the as-run version saved a bare state_dict that
# the resume branch could not read), and the RNG states are saved and restored so that a resumed run continues the
# same random stream. The four independent-seed training logs of this study ran without resuming (epochs 1 to 200
# consecutive), so no published checkpoint was affected.
from release_utils import resume_start_epoch


def rng_state():
    return {'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'numpy': np.random.get_state(), 'python': random.getstate()}


def set_rng_state(st):
    torch.set_rng_state(st['torch'])
    if st.get('cuda') is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st['cuda'])
    np.random.set_state(st['numpy']); random.setstate(st['python'])


def checkpoint_dict(epoch_done):
    return {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
            'epoch': epoch_done, 'rng': rng_state()}


start_epoch, last_ckpt = resume_start_epoch(SAVE_DIR, EPOCHS)
if last_ckpt:
    print(f"Resuming from {last_ckpt} ({start_epoch} epochs completed)")
    ckpt = torch.load(last_ckpt, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['scheduler'])
    if 'rng' in ckpt:
        set_rng_state(ckpt['rng'])
else:
    torch.save(checkpoint_dict(0), os.path.join(SAVE_DIR, 'checkpoint_000.pt'))
    print("Saved random init as checkpoint_000.pt")


def accuracy(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            correct += (model(x).argmax(1) == y).sum().item()
            total += len(y)
    return correct / total


print(f"\n{'ep':>4s}  {'lr':>8s}  {'loss':>8s}  {'tr_acc':>8s}  {'te_acc':>8s}  {'gap':>8s}  {'time':>6s}")
print('-' * 60)
for epoch in range(start_epoch, EPOCHS):
    model.train()
    t0 = time.time()
    running_loss = 0.0
    n_batches = 0
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        n_batches += 1
    scheduler.step()
    avg_loss = running_loss / n_batches
    tr_acc = accuracy(model, train_loader)
    te_acc = accuracy(model, test_loader)
    gap = tr_acc - te_acc
    dt = time.time() - t0
    lr_now = optimizer.param_groups[0]['lr']
    print(f"{epoch+1:4d}  {lr_now:8.5f}  {avg_loss:8.4f}  {tr_acc:8.4f}  {te_acc:8.4f}  {gap:+8.4f}  {dt:6.1f}s",
          flush=True)
    ckpt_path = os.path.join(SAVE_DIR, f'checkpoint_{epoch+1:03d}.pt')
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'rng': rng_state(),
        'epoch': epoch + 1,
        'tr_acc': tr_acc,
        'te_acc': te_acc,
    }, ckpt_path)

print("\nDone.")
