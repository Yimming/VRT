"""Training recipes for ResNet-50, regularization controls and SimCLR.

SimCLR loss and training utilities derive from SupContrast, copyright 2020
Yonglong Tian, BSD-2-Clause; see third_party/SupContrast/LICENSE.
"""
import argparse
import csv
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

from common import parse_range, resume_start_epoch
from models import MEAN_STD, SupConResNet, load_ckpt


def train_r50(args):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    DATA = args.data_root
    SAVE_DIR = args.out
    os.makedirs(SAVE_DIR, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    EPOCHS = 200
    BATCH = 128
    LR = 0.1
    MOMENTUM = 0.9
    WD = 0.0005
    MILESTONES = [60, 120, 160]
    GAMMA = 0.2
    train_tfm = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])])
    test_tfm = T.Compose([T.ToTensor(), T.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])])
    train_ds = torchvision.datasets.CIFAR100(DATA, True, download=True, transform=train_tfm)
    test_ds = torchvision.datasets.CIFAR100(DATA, False, download=True, transform=test_tfm)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)
    model = torchvision.models.resnet50(weights=None, num_classes=100)
    model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    model.maxpool = nn.Identity()
    model = model.to(DEVICE)
    print(f'Model: ResNet50 for CIFAR-100, D=2048, seed={args.seed}')
    print(f'Params: {sum((p.numel() for p in model.parameters())):,}')
    print(f'Device: {DEVICE}')
    print(f'Save: {SAVE_DIR}')
    print(f'Config: epochs={EPOCHS}, batch={BATCH}, lr={LR}, wd={WD}')
    print(f'Schedule: MultiStepLR milestones={MILESTONES}, gamma={GAMMA}')
    print(f'Train: {len(train_ds)}, Test: {len(test_ds)}\n')
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES, gamma=GAMMA)
    criterion = nn.CrossEntropyLoss()
    def rng_state():
        return {'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None, 'numpy': np.random.get_state(), 'python': random.getstate()}
    def set_rng_state(st):
        torch.set_rng_state(st['torch'])
        if st.get('cuda') is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(st['cuda'])
        np.random.set_state(st['numpy'])
        random.setstate(st['python'])
    def checkpoint_dict(epoch_done):
        return {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'epoch': epoch_done, 'rng': rng_state()}
    start_epoch, last_ckpt = resume_start_epoch(SAVE_DIR, EPOCHS)
    if last_ckpt:
        print(f'Resuming from {last_ckpt} ({start_epoch} epochs completed)')
        ckpt = torch.load(last_ckpt, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        if 'rng' in ckpt:
            set_rng_state(ckpt['rng'])
    else:
        torch.save(checkpoint_dict(0), os.path.join(SAVE_DIR, 'checkpoint_000.pt'))
        print('Saved random init as checkpoint_000.pt')
    def accuracy(model, loader):
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = (x.to(DEVICE), y.to(DEVICE))
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
            x, y = (x.to(DEVICE), y.to(DEVICE))
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
        print(f'{epoch + 1:4d}  {lr_now:8.5f}  {avg_loss:8.4f}  {tr_acc:8.4f}  {te_acc:8.4f}  {gap:+8.4f}  {dt:6.1f}s', flush=True)
        ckpt_path = os.path.join(SAVE_DIR, f'checkpoint_{epoch + 1:03d}.pt')
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'rng': rng_state(), 'epoch': epoch + 1, 'tr_acc': tr_acc, 'te_acc': te_acc}, ckpt_path)
    print('\nDone.')

def train_regularization(args):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    DATA = args.data_root
    SAVE_ROOT = args.out
    os.makedirs(SAVE_ROOT, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    configs = {'A_baseline': {'wd': 0.0005, 'aug': False}, 'B_strong_wd': {'wd': 0.005, 'aug': False}, 'C_strong_aug': {'wd': 0.0005, 'aug': True}, 'D_wd_plus_aug': {'wd': 0.005, 'aug': True}}
    configs = {k: v for k, v in configs.items() if k in args.configs.split(',')}
    norm = T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.2435, 0.2616))
    tfm_basic = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), norm])
    tfm_strong = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.AutoAugment(T.AutoAugmentPolicy.CIFAR10), T.ToTensor(), norm, T.RandomErasing(p=0.25)])
    tfm_test = T.Compose([T.ToTensor(), norm])
    te_loader = DataLoader(torchvision.datasets.CIFAR10(DATA, False, download=False, transform=tfm_test), batch_size=512, shuffle=False, num_workers=4, pin_memory=True)
    for name, cfg in configs.items():
        print('=' * 60)
        print(f"Training: {name} (wd={cfg['wd']}, aug={cfg['aug']}, seed={args.seed})")
        print('=' * 60)
        save_dir = os.path.join(SAVE_ROOT, name)
        os.makedirs(save_dir, exist_ok=True)
        if os.path.exists(os.path.join(save_dir, 'checkpoint_049.pt')):
            print('  Already done, skip.')
            continue
        tfm_tr = tfm_strong if cfg['aug'] else tfm_basic
        tr_loader = DataLoader(torchvision.datasets.CIFAR10(DATA, True, download=False, transform=tfm_tr), batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
        model = torchvision.models.resnet18(weights=None, num_classes=10)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model = model.to(DEVICE)
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=cfg['wd'])
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, epochs=50, steps_per_epoch=len(tr_loader))
        criterion = nn.CrossEntropyLoss()
        json.dump({**cfg, 'seed': args.seed}, open(os.path.join(save_dir, 'config.json'), 'w'))
        log = []
        for ep in range(50):
            t0 = time.time()
            model.train()
            tr_loss, tr_correct, tr_total = (0, 0, 0)
            for x, y in tr_loader:
                x, y = (x.to(DEVICE), y.to(DEVICE))
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
            te_correct, te_total, te_loss = (0, 0, 0)
            with torch.no_grad():
                for x, y in te_loader:
                    x, y = (x.to(DEVICE), y.to(DEVICE))
                    out = model(x)
                    te_loss += criterion(out, y).item() * len(y)
                    te_correct += (out.argmax(1) == y).sum().item()
                    te_total += len(y)
            tr_acc = tr_correct / tr_total
            te_acc = te_correct / te_total
            dt = time.time() - t0
            log.append({'ep': ep, 'tr_loss': tr_loss / tr_total, 'te_loss': te_loss / te_total, 'tr_acc': tr_acc, 'te_acc': te_acc, 'gap': tr_acc - te_acc})
            torch.save(model.state_dict(), os.path.join(save_dir, f'checkpoint_{ep:03d}.pt'))
            print(f'  ep{ep:3d}: tr_acc={tr_acc:.4f} te_acc={te_acc:.4f} gap={tr_acc - te_acc:+.4f} ({dt:.0f}s)', flush=True)
        with open(os.path.join(save_dir, 'progress.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=log[0].keys())
            w.writeheader()
            w.writerows(log)
        print(f'  Saved to {save_dir}\n')
    print('\nAll training done.')

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = torch.device('cuda') if features.is_cuda else torch.device('cpu')
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0)
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-06, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss

class TwoCropTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * args.lr_decay_rate ** 3
        lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = sum((epoch > e for e in args.lr_decay_epochs))
        if steps > 0:
            lr = lr * args.lr_decay_rate ** steps
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def parse_simclr(argv=None):
    parser = argparse.ArgumentParser('SimCLR training (SupContrast-faithful)')
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=0.5)
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'])
    parser.add_argument('--size', type=int, default=32)
    parser.add_argument('--temp', type=float, default=0.5)
    parser.add_argument('--cosine', action='store_true', default=True)
    parser.add_argument('--trial', type=str, default='0')
    parser.add_argument('--seed', type=int, default=0, help='random seed (init + augmentation sampling). NOTE: cudnn.benchmark stays on for speed, so runs are seed-controlled but not bitwise-deterministic.')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--data-root', '--data_folder', dest='data_folder', required=True)
    parser.add_argument('--out', '--save_root', dest='save_root', required=True)
    opt = parser.parse_args(argv)
    opt.lr_decay_epochs = [int(it) for it in opt.lr_decay_epochs.split(',')]
    opt.model_name = 'SimCLR_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}'.format(opt.dataset, opt.model, opt.learning_rate, opt.weight_decay, opt.batch_size, opt.temp, opt.trial)
    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)
    opt.warm = opt.batch_size > 256
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * opt.lr_decay_rate ** 3
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate
    opt.save_folder = os.path.join(opt.save_root, opt.model_name)
    os.makedirs(opt.save_folder, exist_ok=True)
    return opt

def set_loader(opt):
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.201)
    else:
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform = transforms.Compose([transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.0)), transforms.RandomHorizontalFlip(), transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8), transforms.RandomGrayscale(p=0.2), transforms.ToTensor(), normalize])
    cls = datasets.CIFAR10 if opt.dataset == 'cifar10' else datasets.CIFAR100
    train_dataset = cls(root=opt.data_folder, transform=TwoCropTransform(train_transform), download=True)
    return torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, pin_memory=True)

def set_model(opt):
    model = SupConResNet(name=opt.model)
    criterion = SupConLoss(temperature=opt.temp)
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        torch.backends.cudnn.benchmark = True
    return (model, criterion)

def train(train_loader, model, criterion, optimizer, epoch, opt):
    model.train()
    batch_time, data_time, losses = (AverageMeter(), AverageMeter(), AverageMeter())
    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)
        images = torch.cat([images[0], images[1]], dim=0)
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
        bsz = labels.shape[0]
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)
        features = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        loss = criterion(features)
        losses.update(loss.item(), bsz)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\tBT {batch_time.val:.3f} ({batch_time.avg:.3f})\tloss {loss.val:.3f} ({loss.avg:.3f})'.format(epoch, idx + 1, len(train_loader), batch_time=batch_time, loss=losses))
            sys.stdout.flush()
    return losses.avg

def train_simclr(argv=None):
    opt = parse_simclr(argv)
    print('save_folder:', opt.save_folder)
    print('seed:', opt.seed)
    import random
    import numpy as np
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(opt.seed)
    train_loader = set_loader(opt)
    model, criterion = set_model(opt)
    optimizer = torch.optim.SGD(model.parameters(), lr=opt.learning_rate, momentum=opt.momentum, weight_decay=opt.weight_decay)
    start_epoch = 1
    last_path = os.path.join(opt.save_folder, 'last.pth')
    if opt.resume and os.path.isfile(last_path):
        state = torch.load(last_path, map_location='cuda' if torch.cuda.is_available() else 'cpu', weights_only=False)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        start_epoch = state['epoch'] + 1
        print(f"resumed from {last_path} (epoch {state['epoch']})")
    log_path = os.path.join(opt.save_folder, 'train_log.csv')
    if start_epoch == 1:
        with open(log_path, 'w', newline='') as f:
            csv.writer(f).writerow(['epoch', 'loss', 'lr', 'seconds'])
    for epoch in range(start_epoch, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)
        t1 = time.time()
        loss = train(train_loader, model, criterion, optimizer, epoch, opt)
        t2 = time.time()
        lr_now = optimizer.param_groups[0]['lr']
        eta_h = (opt.epochs - epoch) * (t2 - t1) / 3600
        print('epoch {}, loss {:.4f}, lr {:.5f}, time {:.1f}s, ETA {:.1f}h'.format(epoch, loss, lr_now, t2 - t1, eta_h))
        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, f'{loss:.5f}', f'{lr_now:.6f}', f'{t2 - t1:.1f}'])
        save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{}.pth'.format(epoch))
        torch.save({'model': model.state_dict(), 'epoch': epoch}, save_file)
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}, last_path)
    print('done.')


def build_loss_views(dataset, train_split, indices, data_root):
    """Fixed augmented pairs for the SimCLR loss trajectories in Appendix F."""
    mean, std = MEAN_STD[dataset]
    transform = T.Compose([
        T.RandomResizedCrop(size=32, scale=(0.2, 1.0)), T.RandomHorizontalFlip(),
        T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        T.RandomGrayscale(p=0.2), T.ToTensor(), T.Normalize(mean, std),
    ])
    cls = torchvision.datasets.CIFAR10 if dataset == "cifar10" else torchvision.datasets.CIFAR100
    dataset_object = cls(root=data_root, train=train_split,
                         transform=TwoCropTransform(transform), download=False)
    torch.manual_seed(0 if train_split else 1)
    first, second = [], []
    for index in indices:
        a, b = dataset_object[int(index)][0]
        first.append(a)
        second.append(b)
    return torch.stack(first), torch.stack(second)


@torch.no_grad()
def mean_contrastive_loss(model, first, second, device, batch_size=512):
    """SupContrast loss, temperature 0.5, averaging complete fixed-order batches."""
    n = (len(first) // batch_size) * batch_size
    if n == 0 or len(first) != len(second):
        raise ValueError("Paired views must have equal length and contain a complete batch")
    losses = []
    for start in range(0, n, batch_size):
        x = torch.cat([first[start:start + batch_size], second[start:start + batch_size]]).to(device, non_blocking=True)
        projection = model(x)
        features = torch.stack(torch.split(projection, batch_size), dim=1)
        losses.append(float(SupConLoss(temperature=0.5)(features)))
    return float(np.mean(losses))


def evaluate_loss(argv=None):
    parser = argparse.ArgumentParser(description="Appendix F: fixed-view SimCLR train/test loss in both BatchNorm modes.")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint-format", required=True, help="Checkpoint path with {ep}, e.g. ckpt_epoch_{ep}.pth")
    parser.add_argument("--epochs", required=True)
    parser.add_argument("--out", required=True, help="New output CSV; existing files are not overwritten")
    parser.add_argument("--model", default="resnet50")
    args = parser.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = sorted(set(parse_range(args.epochs)))
    if not epochs:
        parser.error("At least one epoch is required")
    for epoch in epochs:
        if not os.path.isfile(args.checkpoint_format.format(ep=epoch)):
            raise FileNotFoundError(args.checkpoint_format.format(ep=epoch))
    if os.path.exists(args.out):
        raise FileExistsError(args.out)
    train_indices = np.random.RandomState(0).permutation(50000)[:10000]
    tr1, tr2 = build_loss_views(args.dataset, True, train_indices, args.data_root)
    te1, te2 = build_loss_views(args.dataset, False, np.arange(10000), args.data_root)
    model = SupConResNet(name=args.model).to(device)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "x", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(["epoch", "tr_eval", "te_eval", "gap_eval", "tr_train", "te_train", "gap_train"])
        for epoch in epochs:
            load_ckpt(model, args.checkpoint_format.format(ep=epoch), device)
            model.eval()
            tr_eval = mean_contrastive_loss(model, tr1, tr2, device)
            te_eval = mean_contrastive_loss(model, te1, te2, device)
            model.train()
            tr_train = mean_contrastive_loss(model, tr1, tr2, device)
            te_train = mean_contrastive_loss(model, te1, te2, device)
            values = [tr_eval, te_eval, te_eval - tr_eval, tr_train, te_train, te_train - tr_train]
            if not all(np.isfinite(value) for value in values):
                raise FloatingPointError(f"Non-finite contrastive loss at epoch {epoch}")
            writer.writerow([epoch, *[f"{value:.6f}" for value in values]])
            target.flush()
            print(f"epoch {epoch}: eval gap={te_eval - tr_eval:+.6f}, train-mode gap={te_train - tr_train:+.6f}", flush=True)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    parser = argparse.ArgumentParser(description="Paper training recipes; checkpoints and data are not bundled.")
    sub = parser.add_subparsers(dest="recipe", required=True)
    for name, help_text in (("r50", "ResNet-50 on CIFAR-100, 200 epochs"),
                            ("regularization", "ResNet-18 CIFAR-10 configurations A–D, 50 epochs")):
        child = sub.add_parser(name, help=help_text)
        child.add_argument("--data-root", required=True)
        child.add_argument("--out", required=True)
        child.add_argument("--seed", type=int, required=True)
        if name == "regularization":
            child.add_argument("--configs", default="A_baseline,B_strong_wd,C_strong_aug,D_wd_plus_aug")
    sub.add_parser("simclr", help="SimCLR on CIFAR-10 or CIFAR-100, 1,000 epochs", add_help=False)
    sub.add_parser("evaluate-loss", help="Appendix F: held-out SimCLR loss trajectories", add_help=False)
    if argv and argv[0] == "evaluate-loss":
        return evaluate_loss(argv[1:])
    if argv and argv[0] == "simclr":
        return train_simclr(argv[1:])
    args = parser.parse_args(argv)
    return train_r50(args) if args.recipe == "r50" else train_regularization(args)


if __name__ == "__main__":
    main()
