"""
cifar_null_vrt.py — Type-I check of VRT on CIFAR images through randomly initialized networks.

Every repetition r draws all three sources of randomness anew, so the repetitions are independent null tests:
  network   torch.manual_seed(seed0 + r): a fresh model of the chosen architecture, default initialization, no checkpoint
  split     the 60,000 CIFAR images (official train + test) are split at random into 50,000 "train" and 10,000 "test"
            images (numpy PCG64 generator, seed seed0 + r). Under a random split the two samples are exchangeable, so the
            null hypothesis holds exactly; VRT is a permutation test, so it should reject in about 5 percent of
            repetitions (16 to 35 of 500 is the central 95 percent range of Binomial(500, 0.05)).
  VRT draw  reference and training queries drawn inside VRT with seed vrt_seed0 + r (RandomState, as in every VRT run)
VRT at both budgets, exactly as vrt_forward_budget.py (features L2-normalized in float64, 2,000 permutations):
  large  M = 2,000 reference, 8,000 training queries vs all 10,000 test images, k = 60   (vrt_low_lib.vrt_full, Table 1)
  small  M = 500 reference, 1,000 training vs 1,500 test queries, k = 15                (vrt_low_lib.vrt_low)
The same network is also tested on the OFFICIAL split (official 50,000 train vs official 10,000 test, same VRT seed). There
the 10,000 test images are the same in every repetition ("fixed images, new network, new draw of training images"), so
those repetitions share one test sample and are not independent; the row is a comparison, not the Type-I estimate.

Preprocessing as the paper's feature extraction: ToTensor + Normalize with the statistics of extract_feats.py (SimCLR
encoder: the SupContrast statistics of supcon_fill_vrt.py), no augmentation, fixed batches of 512 (SimCLR 1,024).
  --bn_mode eval   BatchNorm keeps its initial running statistics (mean 0, variance 1), i.e. it does not normalize; this is
                   the extraction mode of Table 1. Each feature depends on its own image only, so one pass over the 60,000
                   images serves both splits.
  --bn_mode train  per-batch statistics (the Section 4.4 extraction). As in the paper, "train" and "test" images go through
                   the network in separate batches, so every split is extracted in its own passes (official and random).
ResNet20-FRN has no BatchNorm (both modes give the same features): run it in eval mode only.

Architectures (--arch): r18_c10, r18_c100 (ResNet-18 of the zoos and Configurations A to D), r50_c100 (ResNet-50 and Phase
Zoo), r50_c10, frn_c10 (ResNet20-FRN), simclr_c10, simclr_c100 (SupContrast ResNet-50 encoder of the SimCLR runs).

Output: <out_dir>/vrt_null_<arch>_<bn_mode>_r<lo>-<hi>.csv, four rows per repetition (split resplit/official x budget):
  arch,bn_mode,rep,net_seed,split_seed,vrt_seed,split,budget,D,dc,p,cos,feat_sec
  (cos = mean cosine between 1,000 train and 1,000 test features of that split, a check for collapsed random features)
Resume: a repetition whose four rows are in any vrt_null_<arch>_<bn_mode>_*.csv of out_dir is skipped; the four rows of a
repetition are written together, so a killed task leaves no half repetition.
Usage (GPU node; job_s50_gpu.sh runs one line of the task file written by cifar_null_vrt_tasks.sh):
  python cifar_null_vrt.py --arch r18_c10 --reps 0-99 --out_dir /home3/p318770/venvs/overfitting/benchmark_final/s50/null_cifar_vrt
"""
import argparse, copy, csv, glob, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torchvision

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vrt_low_lib import vrt_full, vrt_low, l2n
from extract_feats import DATA, C10_STATS, C100_STATS, ResNet20FRN

N_TR, N_TE = 50000, 10000
BUDGETS = {"large": dict(k=60, M=2000, n_qtr=8000), "small": dict(k=15, M=500, n_qtr=1000, n_qte=1500)}
ARCHS = {  # arch: (dataset, normalization statistics, batch size)
    "r18_c10": ("cifar10", "c10", 512), "r18_c100": ("cifar100", "c100", 512),
    "r50_c100": ("cifar100", "c100", 512), "r50_c10": ("cifar10", "c10", 512),
    "frn_c10": ("cifar10", "c10", 512),
    "simclr_c10": ("cifar10", "supcon", 1024), "simclr_c100": ("cifar100", "supcon", 1024),
}


def parse_range(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def stats_for(arch):
    dataset, which, _ = ARCHS[arch]
    if which == "supcon":
        from supcon_fill_vrt import MEAN_STD          # the statistics of the SimCLR feature extraction
        return MEAN_STD[dataset]
    return C10_STATS if which == "c10" else C100_STATS


def load_cifar(arch, dry_run=False):
    """All 60,000 images as one normalized float32 tensor, official order: 50,000 train, then 10,000 test.
    Same values as T.Compose([T.ToTensor(), T.Normalize(*stats)]) image by image."""
    dataset = ARCHS[arch][0]
    if dry_run:                                       # pipeline test only: random images, never for results
        raw = np.random.default_rng(0).integers(0, 256, (N_TR + N_TE, 32, 32, 3), dtype=np.uint8)
    else:
        cls = torchvision.datasets.CIFAR10 if dataset == "cifar10" else torchvision.datasets.CIFAR100
        tr, te = cls(DATA, train=True, download=False), cls(DATA, train=False, download=False)
        assert len(tr.data) == N_TR and len(te.data) == N_TE, (len(tr.data), len(te.data))
        raw = np.concatenate([tr.data, te.data])
    mean, std = stats_for(arch)
    x = torch.from_numpy(raw).permute(0, 3, 1, 2).contiguous().float().div(255)          # T.ToTensor
    return (x - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(std).view(1, 3, 1, 1)  # T.Normalize


def fwd_torchvision(m, x):           # extract_feats.extract_torchvision_resnet without the loop
    x = m.conv1(x); x = m.bn1(x); x = m.relu(x); x = m.maxpool(x)
    x = m.layer1(x); x = m.layer2(x); x = m.layer3(x); x = m.layer4(x)
    return m.avgpool(x).flatten(1)


def fwd_features(m, x):              # ResNet20-FRN, as extract_feats.extract_v7
    return m.forward_features(x)


def fwd_supcon(m, x):                # SimCLR backbone h = model.encoder(x), as supcon_fill_vrt.extract
    return m.encoder(x)


def build(arch):
    """Fresh model with default initialization, built exactly as for the trained checkpoints (CIFAR stem)."""
    if arch[:3] in ("r18", "r50"):
        nc = 100 if arch.endswith("c100") else 10
        fn = torchvision.models.resnet18 if arch.startswith("r18") else torchvision.models.resnet50
        m = fn(weights=None, num_classes=nc)
        m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False); m.maxpool = nn.Identity()
        return m, fwd_torchvision
    if arch == "frn_c10":
        return ResNet20FRN(num_classes=10), fwd_features
    if arch.startswith("simclr"):
        from train_simclr_supcon import SupConResNet
        return SupConResNet(name="resnet50"), fwd_supcon
    raise ValueError(arch)


def set_bn(model, bn_mode):
    model.eval()
    if bn_mode == "train":
        for mod in model.modules():
            if isinstance(mod, nn.modules.batchnorm._BatchNorm):
                mod.train()


@torch.no_grad()
def extract(model, fwd, X, order, bs, device):
    """Features of the images X[order], passed in this order in batches of bs (batch composition matters in train mode)."""
    idx = torch.as_tensor(np.asarray(order), dtype=torch.long)
    out = []
    for s in range(0, len(idx), bs):
        out.append(fwd(model, X[idx[s:s + bs].to(X.device)].to(device, non_blocking=True)).float().cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=list(ARCHS), required=True)
    ap.add_argument("--bn_mode", choices=["eval", "train"], default="eval")
    ap.add_argument("--reps", required=True, help="repetitions, e.g. 0-99")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_perm", type=int, default=2000)
    ap.add_argument("--seed0", type=int, default=1000, help="network and split seed of repetition r: seed0 + r")
    ap.add_argument("--vrt_seed0", type=int, default=100000, help="VRT seed of repetition r: vrt_seed0 + r")
    ap.add_argument("--dry_run", action="store_true", help="random images instead of CIFAR (pipeline test, file tagged DRYRUN)")
    args = ap.parse_args()
    if args.arch == "frn_c10" and args.bn_mode == "train":
        raise SystemExit("ResNet20-FRN has no BatchNorm: train mode would repeat eval mode; run eval only")
    reps = parse_range(args.reps)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bs = ARCHS[args.arch][2]
    tag = f"{args.arch}_{args.bn_mode}" + ("_DRYRUN" if args.dry_run else "")
    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, f"vrt_null_{tag}_r{reps[0]}-{reps[-1]}.csv")

    done = {}                                         # rep -> {(split, budget)}
    for f in glob.glob(os.path.join(args.out_dir, f"vrt_null_{tag}_r*.csv")):
        if os.path.getsize(f) == 0:
            continue
        for r in csv.DictReader(open(f)):
            done.setdefault(int(r["rep"]), set()).add((r["split"], r["budget"]))
    full = {(s, b) for s in ("resplit", "official") for b in BUDGETS}
    todo = [r for r in reps if not full <= done.get(r, set())]
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    print(f"{tag}: {len(todo)} of {len(reps)} repetitions to do ({reps[0]}-{reps[-1]}), device {gpu}, "
          f"{args.n_perm} permutations -> {out_csv}", flush=True)
    if not todo:
        return

    X = load_cifar(args.arch, args.dry_run)
    if device == "cuda":
        X = X.to(device)                              # 737 MB, avoids a host copy per batch
    print(f"images {tuple(X.shape)}  normalized mean {X.mean().item():+.3f}  std {X.std().item():.3f}", flush=True)
    order_all = np.arange(N_TR + N_TE)
    official = (order_all[:N_TR], order_all[N_TR:])

    new = not os.path.exists(out_csv) or os.path.getsize(out_csv) == 0
    fout = open(out_csv, "a", newline=""); w = csv.writer(fout)
    if new:
        w.writerow(["arch", "bn_mode", "rep", "net_seed", "split_seed", "vrt_seed", "split", "budget", "D", "dc", "p",
                    "cos", "feat_sec"]); fout.flush()
    rej = {(s, b): [0, 0] for s in ("resplit", "official") for b in BUDGETS}
    for r in todo:
        t0 = time.time()
        net_seed, split_seed, vrt_seed = args.seed0 + r, args.seed0 + r, args.vrt_seed0 + r
        torch.manual_seed(net_seed)
        model, fwd = build(args.arch)
        model = model.to(device)
        perm = np.random.default_rng(split_seed).permutation(N_TR + N_TE)          # random split of all 60,000 images
        resplit = (perm[N_TE:], perm[:N_TE])                                        # 50,000 "train", 10,000 "test"
        assert len(np.intersect1d(resplit[0], resplit[1])) == 0
        if args.bn_mode == "eval":
            set_bn(model, "eval")
            h = extract(model, fwd, X, order_all, bs, device)
            H = {"resplit": (h[resplit[0]], h[resplit[1]]), "official": (h[:N_TR], h[N_TR:])}
        else:
            sd0 = copy.deepcopy(model.state_dict())
            H = {}
            for name, (a, b) in (("resplit", resplit), ("official", official)):
                parts = []
                for order in (a, b):                  # each split in its own batches, as the paper's two loaders
                    model.load_state_dict(sd0); set_bn(model, "train")
                    parts.append(extract(model, fwd, X, order, bs, device))
                H[name] = tuple(parts)
        t_feat = time.time() - t0
        rows, msg, coss = [], [], {}
        for split in ("resplit", "official"):
            h_tr, h_te = H[split]
            finite = bool(np.isfinite(h_tr).all() and np.isfinite(h_te).all())
            e_tr, e_te = l2n(h_tr.astype(np.float64)), l2n(h_te.astype(np.float64))
            D = e_tr.shape[1]
            cos = coss[split] = float((e_tr[:1000] @ e_te[:1000].T).mean()) if finite else float("nan")
            ps = []
            for budget, kw in BUDGETS.items():
                if not finite:
                    dc, p = float("nan"), float("nan")
                elif budget == "large":
                    dc, p = vrt_full(e_tr, e_te, D, n_perm=args.n_perm, seed=vrt_seed, **kw)
                else:
                    dc, p = vrt_low(e_tr, e_te, D, n_perm=args.n_perm, seed=vrt_seed, **kw)
                rows.append([args.arch, args.bn_mode, r, net_seed, split_seed, vrt_seed, split, budget, D,
                             f"{dc:+.6e}", f"{p:.6f}", f"{cos:.4f}", f"{t_feat:.1f}"])
                if finite:
                    rej[(split, budget)][0] += p < 0.05; rej[(split, budget)][1] += 1
                ps.append(p)
            msg.append(f"{split} p {ps[0]:.3f} / {ps[1]:.3f}" + ("" if finite else " NON-FINITE FEATURES"))
        w.writerows(rows); fout.flush()
        hn = np.median(np.linalg.norm(H["resplit"][0][:2000], axis=1))
        print(f"rep {r:4d}  D {D}  |h| median {hn:.3g}  cos {coss['resplit']:.3f}  {'  '.join(msg)}  "
              f"(features {t_feat:.0f} s, total {time.time() - t0:.0f} s)", flush=True)
        del model, H
    fout.close()
    print("this task, rejections at 0.05 (large / small):  " + "  ".join(
        f"{s} {rej[(s, 'large')][0]}/{rej[(s, 'large')][1]} , {rej[(s, 'small')][0]}/{rej[(s, 'small')][1]}"
        for s in ("resplit", "official")), flush=True)


if __name__ == "__main__":
    main()
