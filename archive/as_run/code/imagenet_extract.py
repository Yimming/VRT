"""
imagenet_extract.py — save the DeepResNet50 ImageNet backbone features used by the paper as npz files
(camera-ready), so that every baseline can be rerun on them with the same scripts as all other settings.

Reproduces benchmark_imagenet_v2.py exactly (model, checkpoint loading, transforms without mean/std
normalization, sorted file order, DataLoader(shuffle=True) with torch.manual_seed(42) for the train split and
torch.manual_seed(123) for the val split, first 50,000 train and 10,000 val images). The only differences:
the features are saved RAW (before F.normalize; every downstream script l2-normalizes, as for ladder_feats),
the dataset returns the image index so the drawn indices are stored alongside, and only the last level
(D = 2,048, the Table 1 rows) is extracted.

Requires the full ImageNet extracted to $TMPDIR/ILSVRC/Data/CLS-LOC (tar xf /scratch/public/ImageNet/imagenet.tar),
1,281,167 training images; the script refuses to run on anything smaller.

Output: /scratch/p318770/model_zoos/ladder_feats/imgnet_<scheme>/ep{ep:03d}.npz with h_tr (50000, 2048) float32,
h_te (10000, 2048) float32, idx_tr, idx_te (indices into the sorted file lists), plus a files_<scheme>.txt manifest.

--bn_mode train (Section 4.4 check): every BatchNorm layer normalizes with the statistics of the current
extraction batch (256 images of one split) instead of the running statistics accumulated during training.
Everything else is unchanged, including the drawn images (same seeds, the script verifies idx_tr/idx_te against
the eval-mode files when they exist). Output goes to imgnet_<scheme>_bntrain/.
"""
import os, sys, time, argparse
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from glob import glob

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CKPT_DIR = '/scratch/p318770/encoder_weights/resnet50'
OUT_ROOT = '/scratch/p318770/model_zoos/ladder_feats'
EPOCHS = [1, 5, 20, 50, 100, 150, 200, 250, 300]
N_TRAIN_FULL = 1281167
N_VAL_FULL = 50000


# ── model: verbatim from benchmark_imagenet_v2.py ─────────────────────────────
class BottleNeckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels//4, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels//4)
        self.conv2 = nn.Conv2d(out_channels//4, out_channels//4, 3, padding=1,
                               stride=2 if downsample else 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels//4)
        self.conv3 = nn.Conv2d(out_channels//4, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.downsample_fn = nn.AvgPool2d(1, stride=2) if downsample else nn.Identity()
        self.need_pad = (out_channels != in_channels)
        self.pad_channels = out_channels - in_channels if self.need_pad else 0

    def forward(self, x):
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.relu(self.bn2(self.conv2(y)))
        y = self.bn3(self.conv3(y))
        skip = self.downsample_fn(x)
        if self.need_pad:
            pad = torch.zeros(skip.shape[0], self.pad_channels,
                              skip.shape[2], skip.shape[3], device=x.device)
            skip = torch.cat([skip, pad], dim=1)
        return self.relu(y + skip)


class DeepResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU()
        )
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
        widths = [3, 4, 6, 3]
        channels = [256, 512, 1024, 2048]
        in_ch = [64, 256, 512, 1024]
        self.levels = nn.ModuleList()
        for i, (w, ic, oc) in enumerate(zip(widths, in_ch, channels)):
            blocks = []
            for j in range(w):
                blocks.append(BottleNeckBlock(
                    ic if j == 0 else oc, oc, downsample=(i > 0 and j == 0)))
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


# ── data: same file order and transforms as benchmark_imagenet_v2.py; label = index ──
class ImageNetVal(Dataset):
    def __init__(self, root, transform):
        self.paths = sorted(glob(os.path.join(root, '*.JPEG')))
        self.transform = transform
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        return self.transform(Image.open(self.paths[i]).convert('RGB')), i


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
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        return self.transform(Image.open(self.paths[i]).convert('RGB')), i


def make_loaders(train_dir, val_dir, bs=256, workers=4):
    tfm = T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor()])   # no Normalize (Hidy did not normalize)
    tr_ds = ImageNetTrain(train_dir, tfm)
    te_ds = ImageNetVal(val_dir, tfm)
    kw = dict(batch_size=bs, num_workers=workers, pin_memory=True)
    return DataLoader(tr_ds, shuffle=True, **kw), DataLoader(te_ds, shuffle=True, **kw)


def set_bn_mode(model, bn_mode):
    """eval: whole network in eval mode (running statistics; paper protocol).
    train: network in eval mode except the BatchNorm layers, which use the statistics of each batch
    (the network has no dropout, so this is the 'train-mode BatchNorm' extraction of Section 4.4)."""
    model.eval()
    n_bn = 0
    if bn_mode == 'train':
        for m in model.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.train(); n_bn += 1
    return n_bn


def extract_raw(model, loader, max_n, data_seed, bn_mode='eval'):
    """Same draw as benchmark_imagenet_v2.extract_embeddings (seed -> shuffled loader -> first max_n),
    but returns the RAW pooled features and the drawn indices."""
    torch.manual_seed(data_seed)
    set_bn_mode(model, bn_mode)
    feats, idxs, n = [], [], 0
    # inference_mode for the paper path; no_grad for train-mode BN, whose in-place running-stat updates
    # are irrelevant here (the checkpoint is reloaded for every epoch) but must be permitted
    ctx = torch.inference_mode() if bn_mode == 'eval' else torch.no_grad()
    with ctx:
        for x, idx in loader:
            feats.append(model.forward_features(x.to(DEVICE), layer='last').float().cpu())
            idxs.append(idx)
            n += x.shape[0]
            if n >= max_n:
                break
    h = torch.cat(feats, 0).numpy()[:max_n].astype(np.float32)
    idx = torch.cat(idxs, 0).numpy()[:max_n]
    return h, idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scheme', required=True, choices=['simple', 'advanced', 'both'])
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--data_root', default=os.path.join(os.environ.get('TMPDIR', '/tmp'), 'ILSVRC', 'Data', 'CLS-LOC'))
    ap.add_argument('--bn_mode', choices=['eval', 'train'], default='eval',
                    help="eval: BatchNorm running statistics (paper). train: per-batch statistics (Section 4.4 check)")
    ap.add_argument('--epochs', default=None, help="comma-separated checkpoint subset, default all nine")
    args = ap.parse_args()
    epochs = [int(e) for e in args.epochs.split(',')] if args.epochs else EPOCHS

    train_dir, val_dir = os.path.join(args.data_root, 'train'), os.path.join(args.data_root, 'val')
    tr_loader, te_loader = make_loaders(train_dir, val_dir, bs=256, workers=args.workers)
    n_tr, n_te = len(tr_loader.dataset), len(te_loader.dataset)
    print(f"device={DEVICE}  bn_mode={args.bn_mode}  train dir {train_dir}: {n_tr} images   val dir {val_dir}: {n_te} images", flush=True)
    if n_tr != N_TRAIN_FULL or n_te != N_VAL_FULL:
        sys.exit(f"REFUSING: expected the full ImageNet ({N_TRAIN_FULL} train / {N_VAL_FULL} val), got {n_tr} / {n_te}")

    schemes = ['simple', 'advanced'] if args.scheme == 'both' else [args.scheme]
    model = DeepResNet50().to(DEVICE)
    print(f"BatchNorm layers switched to batch statistics: {set_bn_mode(model, args.bn_mode)}", flush=True)
    for scheme in schemes:
        base_tag = 'imgnet_simple' if scheme == 'simple' else 'imgnet_adv'
        tag = base_tag + ('_bntrain' if args.bn_mode == 'train' else '')
        out_dir = os.path.join(OUT_ROOT, tag); os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f'files_{scheme}.txt'), 'w') as f:
            f.write("\n".join(tr_loader.dataset.paths) + "\n### val ###\n" + "\n".join(te_loader.dataset.paths) + "\n")
        for ep in epochs:
            out = os.path.join(out_dir, f'ep{ep:03d}.npz')
            if os.path.exists(out):
                print(f"[{tag}] ep{ep:3d} exists, skip", flush=True); continue
            ckpt = os.path.join(CKPT_DIR, f'resnet50_{scheme}_pretraining_checkpoint_epoch_{ep}.pt')
            if not os.path.exists(ckpt):
                print(f"[{tag}] ep{ep:3d} checkpoint not found: {ckpt}", flush=True); continue
            t0 = time.time()
            load_checkpoint(model, ckpt)          # restores weights AND running statistics for every epoch
            h_tr, idx_tr = extract_raw(model, tr_loader, 50000, data_seed=42, bn_mode=args.bn_mode)
            h_te, idx_te = extract_raw(model, te_loader, 10000, data_seed=123, bn_mode=args.bn_mode)
            np.savez(out, h_tr=h_tr, h_te=h_te, idx_tr=idx_tr, idx_te=idx_te)
            pair = ''
            if args.bn_mode == 'train':           # same images as the eval-mode file of this checkpoint?
                ref = os.path.join(OUT_ROOT, base_tag, f'ep{ep:03d}.npz')
                if os.path.exists(ref):
                    r = np.load(ref)
                    same = np.array_equal(r['idx_tr'], idx_tr) and np.array_equal(r['idx_te'], idx_te)
                    pair = '  paired with eval-mode indices: ' + ('OK' if same else 'MISMATCH')
            print(f"[{tag}] ep{ep:3d}  h_tr {h_tr.shape} h_te {h_te.shape}  unique train idx {len(np.unique(idx_tr))}  "
                  f"first idx {idx_tr[:3].tolist()}  ({time.time() - t0:.0f} s){pair}", flush=True)
    print("done")


if __name__ == '__main__':
    main()
