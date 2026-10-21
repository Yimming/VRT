"""
train_simclr_supcon.py — SimCLR training, faithful to HobbitLong/SupContrast
============================================================================
Single-file reimplementation of the SimCLR path of main_supcon.py from
https://github.com/HobbitLong/SupContrast (reference implementation of
Khosla et al. 2020, which officially supports SimCLR). Training semantics
are copied line-for-line from the released code so that the repo's reported
SimCLR-ResNet50 linear-eval numbers (93.6 CIFAR-10 / 70.7 CIFAR-100) remain
citable:

  - augmentation: RandomResizedCrop(32, scale=(0.2,1.0)) + flip
    + ColorJitter(0.4,0.4,0.4,0.1) p=0.8 + grayscale p=0.2 + per-dataset norm
  - two views per image (TwoCropTransform)
  - SupConLoss with temperature=--temp and base_temperature=0.07 (repo default;
    NOTE: this scales the loss by temp/0.07 — intentional, kept verbatim)
  - SGD momentum 0.9, weight_decay 1e-4
  - cosine annealing with eta_min = lr * 0.1^3, adjusted once per epoch
  - per-batch linear warmup for the first 10 epochs, auto-enabled when
    batch_size > 256 (warmup_from=0.01, warmup_to per the repo formula)

Differences from the released main_supcon.py (deliberate, training-neutral):
  - no tensorboard_logger / apex dependency (CSV + stdout logging instead)
  - checkpoints saved to an ABSOLUTE --save_root (never a cwd-relative ./save)
  - intermediate checkpoints store model weights only ({'model', 'epoch'});
    dense early schedule: epochs {1,2,5,15,25,35,45} plus every --save_freq
  - last.pth stores model+optimizer for resume; --resume restarts from it

Self-contained: the model definitions below (BasicBlock ... SupConResNet)
are embedded VERBATIM from SupContrast networks/resnet_big.py, so the
architecture is exactly the repo's.

Usage:
  python train_simclr_supcon.py --dataset cifar10  --trial ssl_c10_s0
  python train_simclr_supcon.py --dataset cifar100 --trial ssl_c100_s0
"""

from __future__ import print_function

import argparse
import csv
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets


# ═══════════════════════════════════════════════════════════════
# model — verbatim from SupContrast networks/resnet_big.py
# ═══════════════════════════════════════════════════════════════
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channel=3, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to:
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, layer=100):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


model_dict = {
    'resnet18': [resnet18, 512],
    'resnet34': [resnet34, 512],
    'resnet50': [resnet50, 2048],
    'resnet101': [resnet101, 2048],
}


class SupConResNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, name='resnet50', head='mlp', feat_dim=128):
        super(SupConResNet, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        feat = self.encoder(x)
        feat = F.normalize(self.head(feat), dim=1)
        return feat


# ═══════════════════════════════════════════════════════════════
# SupConLoss — verbatim from SupContrast losses.py (Yonglong Tian)
# ═══════════════════════════════════════════════════════════════
class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
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

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


# ═══════════════════════════════════════════════════════════════
# utils — verbatim semantics from SupContrast util.py
# ═══════════════════════════════════════════════════════════════
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
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = sum(epoch > e for e in args.lr_decay_epochs)
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


# ═══════════════════════════════════════════════════════════════
# option parsing — mirrors main_supcon.py parse_option()
# ═══════════════════════════════════════════════════════════════
def parse_option():
    parser = argparse.ArgumentParser('SimCLR training (SupContrast-faithful)')

    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=1000)

    parser.add_argument('--learning_rate', type=float, default=0.5)
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)

    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100'])
    parser.add_argument('--size', type=int, default=32)

    parser.add_argument('--temp', type=float, default=0.5)
    parser.add_argument('--cosine', action='store_true', default=True)
    parser.add_argument('--trial', type=str, default='0')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (init + augmentation sampling). '
                             'NOTE: cudnn.benchmark stays on for speed, so '
                             'runs are seed-controlled but not bitwise-'
                             'deterministic.')
    parser.add_argument('--resume', action='store_true')

    parser.add_argument('--data_folder', type=str,
                        default='/home3/p318770/venvs/overfitting')
    parser.add_argument('--save_root', type=str,
                        default='/scratch/p318770/model_zoos/ssl_supcon')

    opt = parser.parse_args()

    opt.lr_decay_epochs = [int(it) for it in opt.lr_decay_epochs.split(',')]

    opt.model_name = 'SimCLR_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}'.\
        format(opt.dataset, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp, opt.trial)
    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training (auto, as in the repo)
    opt.warm = opt.batch_size > 256
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.save_folder = os.path.join(opt.save_root, opt.model_name)
    os.makedirs(opt.save_folder, exist_ok=True)

    return opt


# ═══════════════════════════════════════════════════════════════
# data / model — mirrors set_loader() / set_model()
# ═══════════════════════════════════════════════════════════════
def set_loader(opt):
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    else:
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    cls = datasets.CIFAR10 if opt.dataset == 'cifar10' else datasets.CIFAR100
    train_dataset = cls(root=opt.data_folder,
                        transform=TwoCropTransform(train_transform),
                        download=True)

    return torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True)


def set_model(opt):
    model = SupConResNet(name=opt.model)
    criterion = SupConLoss(temperature=opt.temp)
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        torch.backends.cudnn.benchmark = True
    return model, criterion


# ═══════════════════════════════════════════════════════════════
# train — mirrors train() of main_supcon.py (SimCLR branch)
# ═══════════════════════════════════════════════════════════════
def train(train_loader, model, criterion, optimizer, epoch, opt):
    model.train()
    batch_time, data_time, losses = AverageMeter(), AverageMeter(), AverageMeter()

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
        loss = criterion(features)  # SimCLR: no labels

        losses.update(loss.item(), bsz)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader),
                   batch_time=batch_time, loss=losses))
            sys.stdout.flush()

    return losses.avg


# ═══════════════════════════════════════════════════════════════
EXTRA_EARLY_EPOCHS = (1, 2, 5, 15, 25, 35, 45)


def main():
    opt = parse_option()
    print('save_folder:', opt.save_folder)
    print('seed:', opt.seed)

    # real seeding (--trial is only a name tag)
    import random
    import numpy as np
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(opt.seed)

    train_loader = set_loader(opt)
    model, criterion = set_model(opt)
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=opt.learning_rate,
                                momentum=opt.momentum,
                                weight_decay=opt.weight_decay)

    start_epoch = 1
    last_path = os.path.join(opt.save_folder, 'last.pth')
    if opt.resume and os.path.isfile(last_path):
        state = torch.load(last_path, map_location='cuda' if
                           torch.cuda.is_available() else 'cpu',
                           weights_only=False)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        start_epoch = state['epoch'] + 1
        print(f'resumed from {last_path} (epoch {state["epoch"]})')

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
        print('epoch {}, loss {:.4f}, lr {:.5f}, time {:.1f}s, ETA {:.1f}h'.format(
            epoch, loss, lr_now, t2 - t1, eta_h))
        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, f'{loss:.5f}', f'{lr_now:.6f}',
                                    f'{t2 - t1:.1f}'])

        # intermediate checkpoints: weights only, EVERY epoch
        # (~95 GB per 1000-epoch R50 run; make sure save_root is on /scratch)
        save_file = os.path.join(opt.save_folder,
                                 'ckpt_epoch_{}.pth'.format(epoch))
        torch.save({'model': model.state_dict(), 'epoch': epoch}, save_file)

        # rolling resume point: model + optimizer
        torch.save({'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch}, last_path)

    print('done.')


if __name__ == '__main__':
    main()
