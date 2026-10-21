"""
extract_feats.py — one-time RAW feature extraction for the ladder ablation
==========================================================================
Stage 1 of the ladder/fair-budget experiment (reviewers 9fxN Q1/Q2, nxR4 Q4,
Ftg8 sample-budget). Extracts UNNORMALIZED penultimate features (norms
preserved — the paper's own scripts normalize on the fly and discard them)
for every checkpoint of the three key models and saves fp32 npz files.

Model loading / preprocessing copied verbatim from the supplementary
scripts (r50_multiseed_vrt.py, v7_multiseed_vrt.py,
regularization_multiseed_vrt.py); the ONLY change is that F.normalize is
NOT applied at extraction time.

Stage 2 (ladder_stats.py) is pure NumPy/CPU and consumes these npz files.

Usage (GPU node):
  python extract_feats.py --model r50
  python extract_feats.py --model v7
  python extract_feats.py --model ablA

Output: /scratch/p318770/model_zoos/ladder_feats/{model}/ep{N:03d}.npz
        with arrays h_tr (N_tr x D) and h_te (N_te x D), float32, raw.
"""
import argparse
import json
import os
import pickle
import re as _re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from dataclasses import dataclass
from torch.utils.data import DataLoader

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA = '/home3/p318770/venvs/overfitting'
R50_DIR = '/scratch/p318770/model_zoos/resnet50_cifar100'
V7_DIR = os.path.join(DATA, 'version_7')
REG_ROOT = '/scratch/p318770/model_zoos/ablation_wd_aug'
ZOO_C100 = '/scratch/p318770/model_zoos/tune_zoo_cifar100_resnet18_kaiming_uniform'
ZOO_C10 = '/scratch/p318770/model_zoos/tune_zoo_cifar10_resnet18_kaiming_uniform'
PHASE_ZOO = '/scratch/p318770/model_zoos/phase_transition_zoo/cifar_100-ResNet50_subset'
WP_ROOT = '/scratch/p318770/cnn_wild_park/cnn_wild_park_keep'
WP_TRIAL = 'NN_tune_trainable_2f9ed_00036_36_model=ref_ph_0efa6418_2023-09-24_18-09-57'
PHASE_MODELS = {
    'bs8': 'NN_tune_trainable_bf0b6_00173_173_model_width=64,optim_lr=0.1000,optim_wd=0.0005,seed=0,training_batchsize=8_2024-05-22_15-32-29',
    'bs128': 'NN_tune_trainable_bf0b6_00077_77_model_width=64,optim_lr=0.1000,optim_wd=0.0005,seed=0,training_batchsize=128_2024-05-22_11-30-55',
    'bs1024': 'NN_tune_trainable_bf0b6_00005_5_model_width=64,optim_lr=0.1000,optim_wd=0.0005,seed=0,training_batchsize=1024_2024-05-22_11-30-52',
}
OUT_ROOT = '/scratch/p318770/model_zoos/ladder_feats'

C10_STATS = ([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])
C100_STATS = ([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])


# ── v7 model (verbatim from v7_multiseed_vrt.py) ──────────────────────
class FRN(nn.Module):
    def __init__(self, n, eps=1e-6):
        super().__init__(); self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, n, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, n, 1, 1))
        self.tau = nn.Parameter(torch.zeros(1, n, 1, 1))

    def forward(self, x):
        nu2 = x.pow(2).mean(dim=(2, 3), keepdim=True)
        return torch.maximum(self.gamma * x * torch.rsqrt(nu2 + self.eps) + self.beta, self.tau)


class BasicBlockFRN(nn.Module):
    def __init__(self, inp, out, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(inp, out, 3, stride=stride, padding=1, bias=True); self.bn1 = FRN(out)
        self.conv2 = nn.Conv2d(out, out, 3, padding=1, bias=True); self.bn2 = FRN(out)
        self.act = nn.SiLU(inplace=True); self.shortcut = nn.Sequential()
        if stride != 1 or inp != out:
            self.shortcut = nn.Sequential(nn.Conv2d(inp, out, 1, stride=stride, bias=True), FRN(out))

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x))); out = self.bn2(self.conv2(out))
        return self.act(out + self.shortcut(x))


class ResNet20FRN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__(); self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=True); self.bn1 = FRN(16); self.act = nn.SiLU(inplace=True)
        self.layer1 = self._make(16, 3, 1); self.layer2 = self._make(32, 3, 2); self.layer3 = self._make(64, 3, 2)
        self.linear = nn.Linear(64, num_classes, bias=True)

    def _make(self, planes, n, stride):
        layers = []
        for s in [stride] + [1] * (n - 1):
            layers.append(BasicBlockFRN(self.in_planes, planes, s)); self.in_planes = planes
        return nn.Sequential(*layers)

    def forward_features(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        return F.adaptive_avg_pool2d(x, 1).flatten(1)


# ── WildPark CNN (verbatim from wp_multiseed_vrt.py) ──────────────────
ACTIVATION_FN = {"relu": F.relu, "gelu": F.gelu, "tanh": torch.tanh,
                 "sigmoid": torch.sigmoid, "leaky_relu": F.leaky_relu, "none": lambda x: x}


@dataclass(frozen=True)
class CNNConfig:
    n_layers: int; n_classes: int; channels: list; kernel_size: list
    stride: list; padding: list; residual: list; activation: list


class WPCNN(nn.Module):
    def __init__(self, cfg):
        super().__init__(); self.cfg = cfg; self.layers = nn.ModuleList()
        for i in range(len(cfg.channels) - 1):
            self.layers.append(nn.Conv2d(cfg.channels[i], cfg.channels[i + 1],
                               kernel_size=cfg.kernel_size[i], padding=cfg.padding[i],
                               stride=cfg.stride[i]))
        self.pool = nn.AdaptiveAvgPool2d(1); self.flat = nn.Flatten()
        self.fc = nn.Linear(cfg.channels[-1], cfg.n_classes)

    def forward_features(self, x):
        residuals = {}
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.cfg.residual[i] > -1:
                ch = min(self.cfg.channels[self.cfg.residual[i] + 1], self.cfg.channels[i + 1])
                x = x.clone(); x[:, :ch] += residuals[self.cfg.residual[i]][:, :ch]
            x = ACTIVATION_FN[self.cfg.activation[i]](x)
            if i in self.cfg.residual: residuals[i] = x
        return self.flat(self.pool(x))


def parse_wp_config(params_path):
    params = json.load(open(params_path)); ms = params["model"]
    def ex(name, is_list=False):
        pat = f"{name}=(" + (r"\[.*?\]" if is_list else r"[^,\)]+") + ")"
        v = _re.search(pat, ms).group(1)
        return eval(v) if is_list else int(v) if v.strip().isdigit() else v.strip().strip("'\"")
    return CNNConfig(n_layers=ex("n_layers"), n_classes=ex("n_classes"),
                     channels=ex("channels", True), kernel_size=ex("kernel_size", True),
                     stride=ex("stride", True), padding=ex("padding", True),
                     residual=ex("residual", True), activation=ex("activation", True))


class FakeUnpickler(pickle.Unpickler):
    def find_class(self, m, n):
        try: return super().find_class(m, n)
        except Exception: return type(n, (), {})


def load_wp_ckpt(path):
    fake = type('M', (), {'Unpickler': FakeUnpickler,
                          'load': lambda f, **kw: FakeUnpickler(f).load()})
    return torch.load(path, map_location='cpu', pickle_module=fake)


def find_wp_dir():
    wp_dir = os.path.join(WP_ROOT, WP_TRIAL)
    if not os.path.isdir(wp_dir):
        wp_dir = os.path.join(WP_ROOT, 'cifar10_zooV2', WP_TRIAL)
    if not os.path.isdir(wp_dir):
        for root, dirs, _ in os.walk(WP_ROOT):
            for d in dirs:
                if '2f9ed_00036' in d:
                    return os.path.join(root, d)
    return wp_dir


# ── extraction (RAW: no F.normalize) ──────────────────────────────────
@torch.inference_mode()
def extract_torchvision_resnet(model, loader):
    """Penultimate features of a torchvision resnet (truncated forward),
    exactly as in the supplementary scripts but WITHOUT normalization."""
    model.eval(); out = []
    for x, _ in loader:
        x = x.to(DEVICE)
        x = model.conv1(x); x = model.bn1(x); x = model.relu(x); x = model.maxpool(x)
        x = model.layer1(x); x = model.layer2(x); x = model.layer3(x); x = model.layer4(x)
        out.append(model.avgpool(x).flatten(1).cpu())
    return torch.cat(out, 0).numpy().astype(np.float32)


@torch.inference_mode()
def extract_v7(model, loader):
    model.eval(); out = []
    for x, _ in loader:
        out.append(model.forward_features(x.to(DEVICE)).cpu())
    return torch.cat(out, 0).numpy().astype(np.float32)


def loaders(dataset, stats):
    tfm = T.Compose([T.ToTensor(), T.Normalize(*stats)])
    cls = torchvision.datasets.CIFAR10 if dataset == 'c10' else torchvision.datasets.CIFAR100
    tr = cls(DATA, True, download=False, transform=tfm)
    te = cls(DATA, False, download=False, transform=tfm)
    kw = dict(batch_size=512, shuffle=False, num_workers=4, pin_memory=True)
    return DataLoader(tr, **kw), DataLoader(te, **kw)


def save(out_dir, ep, h_tr, h_te):
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, f'ep{ep:03d}.npz'), h_tr=h_tr, h_te=h_te)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True,
                    choices=['r50', 'v7', 'ablA', 'ablB', 'ablC', 'ablD',
                             'wp', 'phase', 'zoo_c10', 'zoo_c100'])
    ap.add_argument('--submodel', type=int, default=None,
                    help='zoo only: index of a single zoo model to extract')
    ap.add_argument('--r50_dir', default=None,
                    help='override R50 checkpoint dir (e.g. a seed rerun)')
    ap.add_argument('--out_name', default=None,
                    help='override output subdir name under OUT_ROOT')
    args = ap.parse_args()

    if args.model == 'r50':
        tr_l, te_l = loaders('c100', C100_STATS)
        model = torchvision.models.resnet50(weights=None, num_classes=100)
        model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        model.maxpool = nn.Identity()
        model = model.to(DEVICE)
        r50_dir = args.r50_dir or R50_DIR
        out_dir = os.path.join(OUT_ROOT, args.out_name or 'r50')
        for ep in range(0, 201):
            ck = os.path.join(r50_dir, f'checkpoint_{ep:03d}.pt')
            if not os.path.exists(ck):
                continue
            sd = torch.load(ck, map_location='cpu')
            model.load_state_dict(sd['model'] if isinstance(sd, dict) and 'model' in sd else sd)
            save(out_dir, ep, extract_torchvision_resnet(model, tr_l),
                 extract_torchvision_resnet(model, te_l))
            print(f'r50 ep{ep:03d} done', flush=True)

    elif args.model == 'v7':
        from safetensors.torch import load_file
        tr_l, te_l = loaders('c10', C10_STATS)
        model = ResNet20FRN().to(DEVICE)
        out_dir = os.path.join(OUT_ROOT, 'v7')
        for ep in range(0, 200):
            ck = os.path.join(V7_DIR, f'version_7_epoch_{ep}.safetensors')
            if not os.path.exists(ck):
                continue
            model.load_state_dict(load_file(ck), strict=True)
            save(out_dir, ep, extract_v7(model, tr_l), extract_v7(model, te_l))
            print(f'v7 ep{ep:03d} done', flush=True)

    elif args.model == 'wp':
        tr_l, te_l = loaders('c10', C10_STATS)
        wp_dir = find_wp_dir()
        assert os.path.isdir(wp_dir), f'WP dir not found: {wp_dir}'
        cfg = parse_wp_config(os.path.join(wp_dir, 'params.json'))
        model = WPCNN(cfg).to(DEVICE)
        out_dir = os.path.join(OUT_ROOT, 'wp')
        for ep in range(0, 1001, 10):
            ck = os.path.join(wp_dir, f'checkpoint_{ep:06d}', 'checkpoint.pt')
            if not os.path.exists(ck):
                continue
            model.load_state_dict(load_wp_ckpt(ck)['model'])
            save(out_dir, ep, extract_v7(model, tr_l), extract_v7(model, te_l))
            print(f'wp ep{ep:03d} done', flush=True)

    elif args.model == 'phase':
        tr_l, te_l = loaders('c100', C100_STATS)
        model = torchvision.models.resnet50(weights=None, num_classes=100)
        model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        model.maxpool = nn.Identity()
        model = model.to(DEVICE)
        for label, mname in PHASE_MODELS.items():
            out_dir = os.path.join(OUT_ROOT, f'phase_{label}')
            for ep in [100, 125, 150]:
                ck = os.path.join(PHASE_ZOO, mname, f'checkpoint_{ep:06d}', 'checkpoints')
                if not os.path.exists(ck):
                    print(f'phase_{label} ep{ep}: missing'); continue
                sd = torch.load(ck, map_location='cpu')
                sd = {k.replace('module.', ''): v for k, v in sd.items()}
                model.load_state_dict(sd)
                save(out_dir, ep, extract_torchvision_resnet(model, tr_l),
                     extract_torchvision_resnet(model, te_l))
                print(f'phase_{label} ep{ep:03d} done', flush=True)

    elif args.model in ('zoo_c10', 'zoo_c100'):
        c10 = args.model == 'zoo_c10'
        zoo_dir = ZOO_C10 if c10 else ZOO_C100
        n_ep = 51 if c10 else 61
        tr_l, te_l = loaders('c10' if c10 else 'c100',
                             C10_STATS if c10 else C100_STATS)
        subdirs = sorted(d for d in os.listdir(zoo_dir)
                         if os.path.isdir(os.path.join(zoo_dir, d))
                         and not d.startswith('.') and d.startswith('NN_'))
        # paper uses 5 models per zoo; cap in case the local dir holds more
        idxs = [args.submodel] if args.submodel is not None \
            else range(min(5, len(subdirs)))
        model = torchvision.models.resnet18(weights=None,
                                            num_classes=10 if c10 else 100)
        model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        model.maxpool = nn.Identity()
        model = model.to(DEVICE)
        for mi in idxs:
            out_dir = os.path.join(OUT_ROOT, f'{args.model}_m{mi}')
            mpath = os.path.join(zoo_dir, subdirs[mi])
            for ep in range(n_ep):
                ck = os.path.join(mpath, f'checkpoint_{ep:06d}', 'checkpoints')
                if not os.path.exists(ck):
                    continue
                model.load_state_dict(torch.load(ck, map_location='cpu'))
                save(out_dir, ep, extract_torchvision_resnet(model, tr_l),
                     extract_torchvision_resnet(model, te_l))
                print(f'{args.model}_m{mi} ep{ep:03d} done', flush=True)

    else:  # ablA..ablD
        name = {'ablA': 'A_baseline', 'ablB': 'B_strong_wd',
                'ablC': 'C_strong_aug', 'ablD': 'D_wd_plus_aug'}[args.model]
        tr_l, te_l = loaders('c10', C10_STATS)
        model = torchvision.models.resnet18(weights=None, num_classes=10)
        model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        model.maxpool = nn.Identity()
        model = model.to(DEVICE)
        out_dir = os.path.join(OUT_ROOT, args.model)
        for ep in range(0, 50):
            ck = os.path.join(REG_ROOT, name, f'checkpoint_{ep:03d}.pt')
            if not os.path.exists(ck):
                continue
            sd = torch.load(ck, map_location='cpu')
            model.load_state_dict(sd['model'] if isinstance(sd, dict) and 'model' in sd else sd)
            save(out_dir, ep, extract_torchvision_resnet(model, tr_l),
                 extract_torchvision_resnet(model, te_l))
            print(f'{args.model} ep{ep:03d} done', flush=True)

    print('all done ->', out_dir)


if __name__ == '__main__':
    main()
