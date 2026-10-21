"""Backbone models and raw feature extraction for the paper.

The SimCLR model definitions derive from SupContrast, copyright 2020 Yonglong
Tian, BSD-2-Clause; see third_party/SupContrast/LICENSE.
"""
import argparse
import copy
import json
import os
import pickle
import re as _re
from dataclasses import dataclass
from glob import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from common import parse_range

C10_STATS = ([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])
C100_STATS = ([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
MEAN_STD = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
}
N_TR, N_TE = 50000, 10000
N_TRAIN_FULL, N_VAL_FULL = 1281167, 50000


class FRN(nn.Module):

    def __init__(self, n, eps=1e-06):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, n, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, n, 1, 1))
        self.tau = nn.Parameter(torch.zeros(1, n, 1, 1))

    def forward(self, x):
        nu2 = x.pow(2).mean(dim=(2, 3), keepdim=True)
        return torch.maximum(self.gamma * x * torch.rsqrt(nu2 + self.eps) + self.beta, self.tau)

class BasicBlockFRN(nn.Module):

    def __init__(self, inp, out, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(inp, out, 3, stride=stride, padding=1, bias=True)
        self.bn1 = FRN(out)
        self.conv2 = nn.Conv2d(out, out, 3, padding=1, bias=True)
        self.bn2 = FRN(out)
        self.act = nn.SiLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or inp != out:
            self.shortcut = nn.Sequential(nn.Conv2d(inp, out, 1, stride=stride, bias=True), FRN(out))

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + self.shortcut(x))

class ResNet20FRN(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=True)
        self.bn1 = FRN(16)
        self.act = nn.SiLU(inplace=True)
        self.layer1 = self._make(16, 3, 1)
        self.layer2 = self._make(32, 3, 2)
        self.layer3 = self._make(64, 3, 2)
        self.linear = nn.Linear(64, num_classes, bias=True)

    def _make(self, planes, n, stride):
        layers = []
        for s in [stride] + [1] * (n - 1):
            layers.append(BasicBlockFRN(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward_features(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return F.adaptive_avg_pool2d(x, 1).flatten(1)

ACTIVATION_FN = {'relu': F.relu, 'gelu': F.gelu, 'tanh': torch.tanh, 'sigmoid': torch.sigmoid, 'leaky_relu': F.leaky_relu, 'none': lambda x: x}

@dataclass(frozen=True)
class CNNConfig:
    n_layers: int
    n_classes: int
    channels: list
    kernel_size: list
    stride: list
    padding: list
    residual: list
    activation: list

class WPCNN(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList()
        for i in range(len(cfg.channels) - 1):
            self.layers.append(nn.Conv2d(cfg.channels[i], cfg.channels[i + 1], kernel_size=cfg.kernel_size[i], padding=cfg.padding[i], stride=cfg.stride[i]))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(cfg.channels[-1], cfg.n_classes)

    def forward_features(self, x):
        residuals = {}
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.cfg.residual[i] > -1:
                ch = min(self.cfg.channels[self.cfg.residual[i] + 1], self.cfg.channels[i + 1])
                x = x.clone()
                x[:, :ch] += residuals[self.cfg.residual[i]][:, :ch]
            x = ACTIVATION_FN[self.cfg.activation[i]](x)
            if i in self.cfg.residual:
                residuals[i] = x
        return self.flat(self.pool(x))

def parse_wp_config(params_path):
    params = json.load(open(params_path))
    ms = params['model']

    def ex(name, is_list=False):
        pat = f'{name}=(' + ('\\[.*?\\]' if is_list else '[^,\\)]+') + ')'
        v = _re.search(pat, ms).group(1)
        return eval(v) if is_list else int(v) if v.strip().isdigit() else v.strip().strip('\'"')
    return CNNConfig(n_layers=ex('n_layers'), n_classes=ex('n_classes'), channels=ex('channels', True), kernel_size=ex('kernel_size', True), stride=ex('stride', True), padding=ex('padding', True), residual=ex('residual', True), activation=ex('activation', True))

class FakeUnpickler(pickle.Unpickler):

    def find_class(self, m, n):
        try:
            return super().find_class(m, n)
        except Exception:
            return type(n, (), {})

def load_wp_ckpt(path):
    fake = type('M', (), {'Unpickler': FakeUnpickler, 'load': lambda f, **kw: FakeUnpickler(f).load()})
    return torch.load(path, map_location='cpu', pickle_module=fake)

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
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return (out, preact)
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
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return (out, preact)
        else:
            return out

class ResNet(nn.Module):

    def __init__(self, block, num_blocks, in_channel=3, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
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

model_dict = {'resnet18': [resnet18, 512], 'resnet34': [resnet34, 512], 'resnet50': [resnet50, 2048], 'resnet101': [resnet101, 2048]}

class SupConResNet(nn.Module):
    """backbone + projection head"""

    def __init__(self, name='resnet50', head='mlp', feat_dim=128):
        super(SupConResNet, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(nn.Linear(dim_in, dim_in), nn.ReLU(inplace=True), nn.Linear(dim_in, feat_dim))
        else:
            raise NotImplementedError('head not supported: {}'.format(head))

    def forward(self, x):
        feat = self.encoder(x)
        feat = F.normalize(self.head(feat), dim=1)
        return feat

class BottleNeckBlock(nn.Module):

    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels // 4, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels // 4)
        self.conv2 = nn.Conv2d(out_channels // 4, out_channels // 4, 3, padding=1, stride=2 if downsample else 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels // 4)
        self.conv3 = nn.Conv2d(out_channels // 4, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.downsample_fn = nn.AvgPool2d(1, stride=2) if downsample else nn.Identity()
        self.need_pad = out_channels != in_channels
        self.pad_channels = out_channels - in_channels if self.need_pad else 0

    def forward(self, x):
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.relu(self.bn2(self.conv2(y)))
        y = self.bn3(self.conv3(y))
        skip = self.downsample_fn(x)
        if self.need_pad:
            pad = torch.zeros(skip.shape[0], self.pad_channels, skip.shape[2], skip.shape[3], device=x.device)
            skip = torch.cat([skip, pad], dim=1)
        return self.relu(y + skip)

class DeepResNet50(nn.Module):

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False), nn.BatchNorm2d(64), nn.ReLU())
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
        widths = [3, 4, 6, 3]
        channels = [256, 512, 1024, 2048]
        in_ch = [64, 256, 512, 1024]
        self.levels = nn.ModuleList()
        for i, (w, ic, oc) in enumerate(zip(widths, in_ch, channels)):
            blocks = []
            for j in range(w):
                blocks.append(BottleNeckBlock(ic if j == 0 else oc, oc, downsample=i > 0 and j == 0))
            self.levels.append(nn.Sequential(*blocks))

    def forward_features(self, x, layer='last'):
        x = self.stem(x)
        x = self.max_pool(x)
        for i, level in enumerate(self.levels):
            x = level(x)
            if layer == 'second_to_last' and i == 2:
                return F.adaptive_avg_pool2d(x, 1).flatten(1)
        return F.adaptive_avg_pool2d(x, 1).flatten(1)

def load_checkpoint(model, path):
    ckpt = torch.load(path, map_location='cpu')
    new_sd = {k[len('layers.0.'):]: v for k, v in ckpt.items() if k.startswith('layers.0.')}
    model.load_state_dict(new_sd, strict=True)

class ImageNetVal(Dataset):

    def __init__(self, root, transform):
        self.paths = sorted(glob(os.path.join(root, '*.JPEG')))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return (self.transform(Image.open(self.paths[i]).convert('RGB')), i)

class ImageNetTrain(Dataset):

    def __init__(self, root, transform):
        self.transform = transform
        self.paths = []
        for cls_dir in sorted(os.listdir(root)):
            cls_path = os.path.join(root, cls_dir)
            if os.path.isdir(cls_path):
                for f in sorted(os.listdir(cls_path)):
                    if f.endswith('.JPEG'):
                        self.paths.append(os.path.join(cls_path, f))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return (self.transform(Image.open(self.paths[i]).convert('RGB')), i)

def make_loaders(train_dir, val_dir, bs=256, workers=4):
    tfm = T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor()])
    tr_ds = ImageNetTrain(train_dir, tfm)
    te_ds = ImageNetVal(val_dir, tfm)
    kw = dict(batch_size=bs, num_workers=workers, pin_memory=True)
    return (DataLoader(tr_ds, shuffle=True, **kw), DataLoader(te_ds, shuffle=True, **kw))

def set_bn_mode(model, bn_mode):
    """eval: whole network in eval mode (running statistics; paper protocol).
    train: network in eval mode except the BatchNorm layers, which use the statistics of each batch
    (the network has no dropout, so this is the 'train-mode BatchNorm' extraction of Section 4.4)."""
    model.eval()
    n_bn = 0
    if bn_mode == 'train':
        for m in model.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.train()
                n_bn += 1
    return n_bn

ARCHS = {'r18_c10': ('cifar10', 'c10', 512), 'r18_c100': ('cifar100', 'c100', 512), 'r50_c100': ('cifar100', 'c100', 512), 'r50_c10': ('cifar10', 'c10', 512), 'frn_c10': ('cifar10', 'c10', 512), 'simclr_c10': ('cifar10', 'supcon', 1024), 'simclr_c100': ('cifar100', 'supcon', 1024)}

def stats_for(arch):
    dataset, which, _ = ARCHS[arch]
    if which == 'supcon':
        return MEAN_STD[dataset]
    return C10_STATS if which == 'c10' else C100_STATS

def load_cifar(arch, dry_run=False, *, data_root=None):
    """All 60,000 images as one normalized float32 tensor, official order: 50,000 train, then 10,000 test.
    Same values as T.Compose([T.ToTensor(), T.Normalize(*stats)]) image by image."""
    dataset = ARCHS[arch][0]
    if dry_run:
        raw = np.random.default_rng(0).integers(0, 256, (N_TR + N_TE, 32, 32, 3), dtype=np.uint8)
    else:
        cls = torchvision.datasets.CIFAR10 if dataset == 'cifar10' else torchvision.datasets.CIFAR100
        tr, te = (cls(data_root, train=True, download=False), cls(data_root, train=False, download=False))
        assert len(tr.data) == N_TR and len(te.data) == N_TE, (len(tr.data), len(te.data))
        raw = np.concatenate([tr.data, te.data])
    mean, std = stats_for(arch)
    x = torch.from_numpy(raw).permute(0, 3, 1, 2).contiguous().float().div(255)
    return (x - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(std).view(1, 3, 1, 1)

def fwd_torchvision(m, x):
    x = m.conv1(x)
    x = m.bn1(x)
    x = m.relu(x)
    x = m.maxpool(x)
    x = m.layer1(x)
    x = m.layer2(x)
    x = m.layer3(x)
    x = m.layer4(x)
    return m.avgpool(x).flatten(1)

def fwd_features(m, x):
    return m.forward_features(x)

def fwd_supcon(m, x):
    return m.encoder(x)

def build(arch):
    """Fresh model with default initialization, built exactly as for the trained checkpoints (CIFAR stem)."""
    if arch[:3] in ('r18', 'r50'):
        nc = 100 if arch.endswith('c100') else 10
        fn = torchvision.models.resnet18 if arch.startswith('r18') else torchvision.models.resnet50
        m = fn(weights=None, num_classes=nc)
        m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        m.maxpool = nn.Identity()
        return (m, fwd_torchvision)
    if arch == 'frn_c10':
        return (ResNet20FRN(num_classes=10), fwd_features)
    if arch.startswith('simclr'):
        return (SupConResNet(name='resnet50'), fwd_supcon)
    raise ValueError(arch)

def set_bn(model, bn_mode):
    set_bn_mode(model, bn_mode)

@torch.no_grad()
def extract_batch(model, fwd, X, order, bs, device):
    """Features of the images X[order], passed in this order in batches of bs (batch composition matters in train mode)."""
    idx = torch.as_tensor(np.asarray(order), dtype=torch.long)
    out = []
    for s in range(0, len(idx), bs):
        out.append(fwd(model, X[idx[s:s + bs].to(X.device)].to(device, non_blocking=True)).float().cpu())
    return torch.cat(out).numpy()

def load_ckpt(model, path, device):
    state = torch.load(path, map_location=device, weights_only=False)
    sd = state['model']
    sd = {k.replace('encoder.module.', 'encoder.'): v for k, v in sd.items()}
    model.load_state_dict(sd)
    return state.get('epoch')

@torch.no_grad()
def extract(model, dataset, device, *, data_root, workers=8):
    mean, std = MEAN_STD[dataset]
    tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    cls = torchvision.datasets.CIFAR10 if dataset == 'cifar10' else torchvision.datasets.CIFAR100
    model.eval()
    out = {}
    for split, train in (('tr', True), ('te', False)):
        ds = cls(root=data_root, train=train, transform=tf, download=False)
        loader = DataLoader(ds, batch_size=1024, num_workers=workers, pin_memory=True)
        hs, zs = ([], [])
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            h = model.encoder(x)
            z = model(x)
            hs.append(h.float().cpu())
            zs.append(z.float().cpu())
        out[f'h_{split}'] = torch.cat(hs).numpy()
        out[f'z_{split}'] = torch.cat(zs).numpy()
    return out

def bn_train(model):
    set_bn_mode(model, 'train')

@torch.no_grad()
def extract_bntrain(model, ckpt_path, dataset, device, *, data_root, workers=8):
    mean, std = MEAN_STD[dataset]
    tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    cls = torchvision.datasets.CIFAR10 if dataset == 'cifar10' else torchvision.datasets.CIFAR100
    out = {}
    for split, train in (('tr', True), ('te', False)):
        load_ckpt(model, ckpt_path, device)
        bn_train(model)
        ds = cls(root=data_root, train=train, transform=tf, download=False)
        loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=workers, pin_memory=True)
        hs = []
        for x, _ in loader:
            hs.append(model.encoder(x.to(device, non_blocking=True)).float().cpu())
        out[f'h_{split}'] = torch.cat(hs).numpy()
    return out

def extract_raw(model, loader, max_n, data_seed, bn_mode='eval', device=None):
    """Same draw as benchmark_imagenet_v2.extract_embeddings (seed -> shuffled loader -> first max_n),
    but returns the RAW pooled features and the drawn indices."""
    device = device or next(model.parameters()).device
    torch.manual_seed(data_seed)
    set_bn_mode(model, bn_mode)
    feats, idxs, n = ([], [], 0)
    ctx = torch.inference_mode() if bn_mode == 'eval' else torch.no_grad()
    with ctx:
        for x, idx in loader:
            feats.append(model.forward_features(x.to(device), layer='last').float().cpu())
            idxs.append(idx)
            n += x.shape[0]
            if n >= max_n:
                break
    h = torch.cat(feats, 0).numpy()[:max_n].astype(np.float32)
    idx = torch.cat(idxs, 0).numpy()[:max_n]
    return (h, idx)
def load_backbone(model, path, arch, device):
    if arch.startswith("simclr"):
        return load_ckpt(model, path, device)
    if arch == "imagenet":
        return load_checkpoint(model, path)
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state = load_file(path)
    elif arch == "wp":
        state = load_wp_ckpt(path)
    else:
        state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)


def cifar_features(model, forward, arch, data_root, bn_mode="eval", device="cpu", workers=4):
    dataset = "cifar10" if arch == "wp" else ARCHS[arch][0]
    stats = C10_STATS if arch == "wp" else stats_for(arch)
    transform = T.Compose([T.ToTensor(), T.Normalize(*stats)])
    cls = torchvision.datasets.CIFAR10 if dataset == "cifar10" else torchvision.datasets.CIFAR100
    batch_size = 1024 if arch.startswith("simclr") and bn_mode == "eval" else 512
    initial = copy.deepcopy(model.state_dict()) if bn_mode == "train" else None
    result = {}
    for split, is_train in (("tr", True), ("te", False)):
        if initial is not None:
            model.load_state_dict(initial)
        set_bn(model, bn_mode)
        loader = DataLoader(cls(data_root, train=is_train, download=False, transform=transform),
                            batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
        chunks = []
        with torch.no_grad():
            for x, _ in loader:
                chunks.append(forward(model, x.to(device)).float().cpu())
        result["h_" + split] = torch.cat(chunks).numpy()
    return result


def imagenet_features(model, train_loader, test_loader, bn_mode="eval", pair="trval", device=None):
    if pair == "trtr":
        values, indices = extract_raw(model, train_loader, 60000, 42, bn_mode, device)
        result = dict(h_tr=values[:50000], h_te=values[50000:],
                      idx_tr=indices[:50000], idx_te=indices[50000:])
        if len(result["h_te"]) != 10000 or np.intersect1d(result["idx_tr"], result["idx_te"]).size:
            raise ValueError("ImageNet train/train samples must be disjoint and complete")
        return result
    h_tr, idx_tr = extract_raw(model, train_loader, 50000, 42, bn_mode, device)
    h_te, idx_te = extract_raw(model, test_loader, 10000, 123, bn_mode, device)
    return dict(h_tr=h_tr, h_te=h_te, idx_tr=idx_tr, idx_te=idx_te)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Extract raw backbone features for Tables 1–3 and Appendix C.")
    parser.add_argument("--arch", choices=[*ARCHS, "wp", "imagenet"], required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint-format", help="Checkpoint path containing {ep}; not used with --random-init")
    parser.add_argument("--epochs", required=True, help="Epochs or random initialization indices, e.g. 0-29")
    parser.add_argument("--out", required=True, help="Output directory; writes epNNN.npz")
    parser.add_argument("--bn-mode", choices=["eval", "train"], default="eval")
    parser.add_argument("--pair", choices=["trval", "trtr"], default="trval", help="ImageNet only")
    parser.add_argument("--random-init", action="store_true")
    parser.add_argument("--seed0", type=int, default=1000)
    parser.add_argument("--wp-config", help="WildPark params.json")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    if not args.random_init and not args.checkpoint_format:
        parser.error("--checkpoint-format is required unless --random-init is selected")
    if args.arch == "wp" and not args.wp_config:
        parser.error("--wp-config is required for the WildPark model")
    if args.pair == "trtr" and args.arch != "imagenet":
        parser.error("--pair trtr is implemented for ImageNet; CIFAR controls are in controls.py")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    if args.arch == "imagenet":
        tr, te = make_loaders(os.path.join(args.data_root, "train"), os.path.join(args.data_root, "val"), workers=args.workers)
        if (len(tr.dataset), len(te.dataset)) != (N_TRAIN_FULL, N_VAL_FULL):
            raise ValueError("Expected full ImageNet: 1,281,167 train and 50,000 validation images")
    model = None
    for epoch in parse_range(args.epochs):
        output = os.path.join(args.out, f"ep{epoch:03d}.npz")
        if os.path.exists(output):
            continue
        if args.random_init:
            torch.manual_seed(args.seed0 + epoch)
        if model is None or args.random_init:
            if args.arch == "imagenet":
                model = DeepResNet50()
            elif args.arch == "wp":
                model = WPCNN(parse_wp_config(args.wp_config))
                forward = fwd_features
            else:
                model, forward = build(args.arch)
            model = model.to(device)
        if not args.random_init:
            path = args.checkpoint_format.format(ep=epoch)
            if not os.path.isfile(path):
                print(f"missing checkpoint: {path}", flush=True)
                continue
            load_backbone(model, path, args.arch, device)
        if args.arch == "imagenet":
            values = imagenet_features(model, tr, te, args.bn_mode, args.pair, device)
        elif args.arch.startswith("simclr") and args.bn_mode == "eval":
            values = extract(model, ARCHS[args.arch][0], device, data_root=args.data_root, workers=args.workers)
        else:
            values = cifar_features(model, forward, args.arch, args.data_root, args.bn_mode, device, args.workers)
        np.savez(output, **values)
        print(output, flush=True)


if __name__ == "__main__":
    main()
