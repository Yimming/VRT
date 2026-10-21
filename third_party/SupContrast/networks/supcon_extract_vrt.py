"""
supcon_extract_vrt.py — VRT / MMD trajectories on SupContrast SimCLR checkpoints
================================================================================
Place this file NEXT TO train_simclr_supcon.py (imports SupConResNet from it).

For every checkpoint saved by the patched main_supcon.py, extracts train/test
representations in two spaces:
  backbone : h = model.encoder(x)   (2048-d, unnormalized -> l2-normalized here)
  proj     : z = model(x)           (128-d, natively l2-normalized by the head)
and runs the paper's VRT protocol (M=2000, n_qtr=8000, k=60, 2000 perms,
10 seeds) plus per-epoch norm diagnostics on the raw backbone features.

Sanity positive (power check): at the end, VRT is run between the TEST-split
backbone embeddings of the earliest and the final checkpoint. Two different
encoders map the same images to different distributions, so this must reject;
it certifies that a null trajectory reflects absent train-test divergence
rather than absent power in this embedding space.

Usage (from the SupContrast repo root):
  python supcon_extract_vrt.py \
    --ckpt_dir ./save/SupCon/cifar10_models/SimCLR_cifar10_resnet50_lr_0.5_decay_0.0001_bsz_1024_temp_0.5_trial_ssl_c10_s0_cosine_warm \
    --dataset cifar10 --with_mmd
"""

import argparse
import os

import numpy as np
import torch
import torchvision
from torchvision import transforms as T
from torch.utils.data import DataLoader
from scipy.special import betainc, betaln

from train_simclr_supcon import SupConResNet

# ── output dirs ───────────────────────────────────────────────────────
OUT_DIR = '/home3/p318770/venvs/overfitting/benchmark_final'   # small CSVs only
FEATS_ROOT = '/scratch/p318770/model_zoos/ssl_supcon_feats'    # big fp16 dumps
DATA_FOLDER = '/home3/p318770/venvs/overfitting'

MEAN_STD = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
}
SEEDS = list(range(42, 52))   # paper seed convention


# ── VRT (identical to supplementary src/methods.py) ───────────────────
def log_cap_volume(theta, D):
    sin2 = np.sin(theta)**2
    I = betainc((D-1)/2.0, 0.5, sin2)
    V = np.where(theta <= np.pi/2, 0.5*I, 1.0 - 0.5*I)
    safe = V > 1e-300
    log_V = np.empty_like(V)
    log_V[safe] = np.log(V[safe])
    if np.any(~safe):
        a = (D-1) / 2.0
        const = -np.log(2.0) - np.log(a) - betaln(a, 0.5)
        log_V[~safe] = const + (D-1) * np.log(np.maximum(np.sin(theta[~safe]), 1e-300))
    return log_V


def compute_chat(query, ref, D, k, batch_size=1024):
    if k > ref.shape[0]:
        raise ValueError(f"k={k} exceeds reference size M={ref.shape[0]}")
    Nq = query.shape[0]
    theta_all = []
    for s in range(0, Nq, batch_size):
        e = min(s + batch_size, Nq)
        cos = query[s:e] @ ref.T
        pidx = np.argpartition(-cos, k - 1, axis=1)[:, :k]
        top = np.take_along_axis(cos, pidx, axis=1)
        top = np.take_along_axis(top, np.argsort(-top, axis=1), axis=1)
        theta_all.append(np.arccos(np.clip(top, -1, 1)))
    theta = np.concatenate(theta_all, 0)
    L = log_cap_volume(theta, D)
    dL = np.diff(L, axis=1)
    r = np.arange(1, k, dtype=np.float64)
    Z = np.maximum(r[None, :] * dL, 0.0)
    return np.maximum(Z.mean(axis=1), 1e-12)


def vrt_test(emb_tr, emb_te, D, k=60, M=2000, n_qtr=8000, n_perm=2000, seed=42):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(emb_tr))
    ref = emb_tr[perm[:M]]
    q_tr = emb_tr[perm[M:M + min(n_qtr, len(emb_tr) - M)]]
    c_tr = compute_chat(q_tr, ref, D, k)
    c_te = compute_chat(emb_te, ref, D, k)
    obs = np.median(c_te) - np.median(c_tr)
    pooled = np.concatenate([c_tr, c_te])
    na = len(c_tr)
    count = 0
    for _ in range(n_perm):
        p = rng.permutation(len(pooled))
        if abs(np.median(pooled[p[na:]]) - np.median(pooled[p[:na]])) >= abs(obs):
            count += 1
    return obs, (count + 1) / (n_perm + 1)


def mmd_test(emb_tr, emb_te, n_sub=1500, n_perm=1000, seed=42):
    """Gaussian-RBF MMD^2, median heuristic, permutation p. Mathematically
    identical to the supplementary mmd_test, but the pooled kernel is
    computed ONCE and each permutation is a single matvec (~100x faster
    at D=2048; the naive version rebuilds the kernel every permutation)."""
    rng = np.random.RandomState(seed)
    n_x = min(n_sub, len(emb_tr))
    n_y = min(n_sub, len(emb_te))
    X = emb_tr[rng.permutation(len(emb_tr))[:n_x]]
    Y = emb_te[rng.permutation(len(emb_te))[:n_y]]
    pooled = np.concatenate([X, Y]).astype(np.float64)
    n, m = len(X), len(Y)
    N = n + m
    sub = pooled[rng.choice(N, min(2000, N), replace=False)]
    d2s = np.sum(sub**2, 1, keepdims=True) + np.sum(sub**2, 1, keepdims=True).T - 2*sub@sub.T
    d2s = np.maximum(d2s, 0.0)
    np.fill_diagonal(d2s, np.inf)
    vals = d2s[np.isfinite(d2s) & (d2s > 0)]
    sigma = max(float(np.sqrt(np.median(vals))), 1e-8) if len(vals) else 1.0

    sq = np.sum(pooled**2, 1, keepdims=True)
    d2 = np.maximum(sq + sq.T - 2*pooled@pooled.T, 0.0)
    Km = np.exp(-d2 / (2*sigma**2)).astype(np.float32)
    del d2
    np.fill_diagonal(Km, 0.0)
    tot = float(Km.sum())

    def stat(z):
        S = Km @ z
        Kxx = float(z @ S)
        sx = float(S.sum())
        Kyy = tot - 2*sx + Kxx
        Kxy = sx - Kxx
        return Kxx/(n*(n-1)) + Kyy/(m*(m-1)) - 2*Kxy/(n*m)

    z0 = np.zeros(N, dtype=np.float32)
    z0[:n] = 1.0
    obs = stat(z0)
    count = 0
    for _ in range(n_perm):
        z = np.zeros(N, dtype=np.float32)
        z[rng.permutation(N)[:n]] = 1.0
        if stat(z) >= obs:
            count += 1
    return obs, (count + 1) / (n_perm + 1)


# ── helpers ───────────────────────────────────────────────────────────
def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def norm_perm_test(ntr, nte, n_perm=2000, seed=0):
    rng = np.random.RandomState(seed)
    obs = np.median(nte) - np.median(ntr)
    pooled = np.concatenate([ntr, nte])
    na = len(ntr)
    count = 0
    for _ in range(n_perm):
        p = rng.permutation(len(pooled))
        if abs(np.median(pooled[p[na:]]) - np.median(pooled[p[:na]])) >= abs(obs):
            count += 1
    return obs, (count + 1) / (n_perm + 1)


def load_ckpt(model, path, device):
    state = torch.load(path, map_location=device, weights_only=False)
    sd = state["model"]
    sd = {k.replace("encoder.module.", "encoder."): v for k, v in sd.items()}
    model.load_state_dict(sd)
    return state.get("epoch")


@torch.no_grad()
def extract(model, dataset, device):
    mean, std = MEAN_STD[dataset]
    tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    cls = torchvision.datasets.CIFAR10 if dataset == "cifar10" else torchvision.datasets.CIFAR100
    model.eval()
    out = {}
    for split, train in (("tr", True), ("te", False)):
        ds = cls(root=DATA_FOLDER, train=train, transform=tf, download=False)
        loader = DataLoader(ds, batch_size=1024, num_workers=8, pin_memory=True)
        hs, zs = [], []
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            h = model.encoder(x)
            z = model(x)  # natively l2-normalized by SupConResNet head
            hs.append(h.float().cpu())
            zs.append(z.float().cpu())
        out[f"h_{split}"] = torch.cat(hs).numpy()
        out[f"z_{split}"] = torch.cat(zs).numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--dataset", choices=["cifar10", "cifar100"], required=True)
    ap.add_argument("--model", default="resnet50")
    ap.add_argument("--with_mmd", action="store_true")
    ap.add_argument("--every", type=int, default=1,
                    help="process only epochs divisible by N, plus the dense "
                         "early set {1,2,5,15,25,35,45}")
    ap.add_argument("--save_feats", action="store_true",
                    help="also dump raw 2048-d backbone features per checkpoint "
                         "as fp16 npz (~245MB/ckpt) for later reuse (ladder table)")
    ap.add_argument("--mmd_sub", type=int, default=1500)
    ap.add_argument("--k", type=int, default=60)
    ap.add_argument("--M", type=int, default=2000)
    ap.add_argument("--n_qtr", type=int, default=8000)
    ap.add_argument("--n_perm", type=int, default=2000)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUT_DIR, exist_ok=True)
    tag = f"{args.dataset}_{os.path.basename(os.path.normpath(args.ckpt_dir))}"

    names = [f for f in os.listdir(args.ckpt_dir)
             if f.startswith("ckpt_epoch_") and f.endswith(".pth")]
    names.sort(key=lambda s: int(s[len("ckpt_epoch_"):-len(".pth")]))
    early = {1, 2, 5, 15, 25, 35, 45}
    if args.every > 1:
        names = [n for n in names
                 if (lambda e: e % args.every == 0 or e in early)(
                     int(n[len("ckpt_epoch_"):-len(".pth")]))]
    print(f"{len(names)} checkpoints selected in {args.ckpt_dir}")

    # resume support: epochs with a full set of VRT rows (20 = 10 seeds x 2
    # spaces) are considered done and skipped on restart
    vrt_csv = os.path.join(OUT_DIR, f"vrt_{tag}.csv")
    done_eps = set()
    if os.path.exists(vrt_csv):
        counts = {}
        with open(vrt_csv) as f:
            next(f, None)
            for line in f:
                ep0 = int(line.split(",")[0])
                counts[ep0] = counts.get(ep0, 0) + 1
        done_eps = {e for e, c in counts.items() if c >= 20}
        print(f"resume: {len(done_eps)} epochs already complete, skipping")
    else:
        with open(vrt_csv, "w") as f:
            f.write("epoch,space,seed,dc,p\n")
    norm_csv = os.path.join(OUT_DIR, f"norms_{tag}.csv")
    if not os.path.exists(norm_csv):
        with open(norm_csv, "w") as f:
            f.write("epoch,mean_norm_tr,mean_norm_te,std_norm_tr,std_norm_te,dnorm,p_norm\n")
    if args.with_mmd:
        mmd_csv = os.path.join(OUT_DIR, f"mmd_{tag}.csv")
        if not os.path.exists(mmd_csv):
            with open(mmd_csv, "w") as f:
                f.write("epoch,space,seed,p\n")

    model = SupConResNet(name=args.model).to(device)
    first_h_te, last_h_te = None, None

    for name in names:
        ep = int(name[len("ckpt_epoch_"):-len(".pth")])
        if ep in done_eps:
            continue
        load_ckpt(model, os.path.join(args.ckpt_dir, name), device)
        feats = extract(model, args.dataset, device)

        if first_h_te is None:
            first_ep, first_h_te = ep, feats["h_te"].copy()
        last_ep, last_h_te = ep, feats["h_te"]

        if args.save_feats:
            fdir = os.path.join(FEATS_ROOT, f"feats_{tag}")
            os.makedirs(fdir, exist_ok=True)
            np.savez_compressed(
                os.path.join(fdir, f"h_ep{ep:04d}.npz"),
                h_tr=feats["h_tr"].astype(np.float16),
                h_te=feats["h_te"].astype(np.float16))

        ntr = np.linalg.norm(feats["h_tr"], axis=1)
        nte = np.linalg.norm(feats["h_te"], axis=1)
        dnorm, p_norm = norm_perm_test(ntr, nte, seed=0)
        with open(norm_csv, "a") as f:
            f.write(f"{ep},{ntr.mean():.4f},{nte.mean():.4f},"
                    f"{ntr.std():.4f},{nte.std():.4f},{dnorm:+.5f},{p_norm:.5f}\n")

        for space, (etr, ete) in {
            "backbone": (l2n(feats["h_tr"]), l2n(feats["h_te"])),
            "proj": (l2n(feats["z_tr"]), l2n(feats["z_te"])),
        }.items():
            D = etr.shape[1]
            ps = []
            for seed in SEEDS:
                dc, p = vrt_test(etr, ete, D, k=args.k, M=args.M,
                                 n_qtr=args.n_qtr, n_perm=args.n_perm, seed=seed)
                ps.append(p)
                with open(vrt_csv, "a") as f:
                    f.write(f"{ep},{space},{seed},{dc:+.6f},{p:.6f}\n")
            print(f"ep {ep:4d} [{space:8s}] median p = {np.median(ps):.4f}  "
                  f"(norm p = {p_norm:.4f})", flush=True)

            if args.with_mmd:
                for seed in SEEDS[:5]:
                    _, p = mmd_test(etr, ete, n_sub=args.mmd_sub, n_perm=1000, seed=seed)
                    with open(mmd_csv, "a") as f:
                        f.write(f"{ep},{space},{seed},{p:.6f}\n")

    # ── sanity positive: earliest vs final checkpoint, test split only ──
    if first_h_te is not None and last_ep != first_ep:
        A, B = l2n(first_h_te), l2n(last_h_te)
        # split A into "train-role" sample (reference+queries) and B as test-role
        dc, p = vrt_test(A, B, A.shape[1], k=args.k, M=args.M,
                         n_qtr=min(args.n_qtr, len(A) - args.M), n_perm=args.n_perm, seed=0)
        print(f"[sanity positive] ep{first_ep} vs ep{last_ep} test-split backbone: "
              f"dc={dc:+.5f}, p={p:.5f}  (must reject; certifies power)")
        with open(os.path.join(OUT_DIR, f"sanity_{tag}.txt"), "w") as f:
            f.write(f"ep{first_ep}_vs_ep{last_ep},dc={dc:+.6f},p={p:.6f}\n")

    print(f"done -> {vrt_csv}")


if __name__ == "__main__":
    main()
