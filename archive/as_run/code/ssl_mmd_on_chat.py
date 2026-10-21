"""
ssl_mmd_on_chat.py — MMD on VRT's c-hat scores for the two SimCLR models (Appendix I extension, October 13).

Same comparison as mmd_on_chat_s50.py for the CIFAR models, but SimCLR has no stored feature files, so every checkpoint
is extracted on the fly exactly as ssl_s50_fill.py does for the Table 1 VRT (eval-mode BatchNorm, 2,048-d encoder output,
l2-normalized and cast to float32). For every analysis seed the c-hat values are those of supcon_fill_vrt.vrt_test with
that seed (M = 2,000 reference, the next 8,000 training points and all 10,000 test points as queries, k = 60), and MMD
(mmd_tst_fast.mmd_test_fast: Gaussian kernel, median heuristic, unbiased, 2,000 permutations, seed = analysis seed) runs on
all 8,000 + 10,000 scalar values.
Check: for --check_seed (default 42) the full VRT p-value of the same draw is computed too (column vrt_p); it must equal
the p of the Table 1 SimCLR VRT file ($S50/ssl/ssl_<dataset>_*_c*.csv, stat vrt, budget b10000) for that epoch and seed.
Not every epoch is run: phase 1 takes a grid (every 10 epochs), phase 2 every epoch between the last non-rejecting grid
epoch and the grid onset (onc_summary.py --brackets writes the phase-2 task lines).
CSV: tag,epoch,seed,dc,mmd_c,p,vrt_p,n_inf   (tag simclr_c10 or simclr_c100; vrt_p empty except for the check seed;
n_inf = number of c-hat values that were +inf and capped before MMD, see cap_infinite; added October 14 after the first
CIFAR-100 run returned NaN statistics with p = 1/2001 for 1,365 of 5,450 rows).
Usage (GPU node, venv active, s50 code folder):
  python ssl_mmd_on_chat.py --ckpt_dir "$(ls -d /scratch/p318770/model_zoos/ssl_supcon/SimCLR_cifar10_* | head -1)" \
      --dataset cifar10 --epochs 10-100/10 --out $S50/mmd_on_chat/mmdchat_simclr_c10_c10-100.csv
"""
import argparse, csv, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from supcon_fill_vrt import SupConResNet, load_ckpt, extract, l2n, vrt_test, compute_chat
from mmd_tst_fast import mmd_test_fast

M, K, N_QTR, N_PERM = 2000, 60, 8000, 2000
TAG = {"cifar10": "simclr_c10", "cifar100": "simclr_c100"}


def parse_epochs(s):
    """'10-100/10' (every 10th), '201-209' (every epoch), '5,7,9', or combinations separated by commas"""
    out = []
    for part in s.split(","):
        step = 1
        if "/" in part:
            part, st = part.split("/"); step = int(st)
        if "-" in part:
            a, b = part.split("-"); out += list(range(int(a), int(b) + 1, step))
        else:
            out.append(int(part))
    return sorted(set(out))


def cap_infinite(c_tr, c_te):
    """c-hat is computed in float32 here, exactly as for the Table 1 VRT. A query that coincides with a reference point
    in float32 (angle 0, e.g. a duplicated CIFAR-100 image) gets c-hat = +inf: the Python-float floor 1e-300 of
    log_cap_volume is cast to the float32 array dtype and becomes 0 (NumPy 1.26, the version of the runs, promotes a
    Python scalar to the array dtype when its magnitude fits; NumPy 2 does the same under NEP 50), so log(0) = -inf.
    The VRT median ignores this (+inf only ranks last), but a kernel on +inf gives inf - inf = NaN, and the as-run
    mmd_test_fast then returned p = 1/(n_perm + 1) (the released version raises instead). Every +inf is replaced by one
    finite value far above all finite c-hat of the draw (largest + 10 x range), which leaves both medians, hence the VRT
    statistic, unchanged. Kernel values of a capped point: close to 0 against every finite point (bandwidth <= range)
    and exactly 1 against another capped point, since all capped points share the same value. This is the policy used
    for the published MMD-on-c-hat numbers; it is kept as run."""
    pooled = np.concatenate([c_tr, c_te])
    if np.isnan(pooled).any() or np.isneginf(pooled).any():
        raise RuntimeError("c-hat contains NaN or -inf")
    n_inf = int(np.isposinf(pooled).sum())
    if n_inf:
        fin = pooled[np.isfinite(pooled)]
        cap = float(fin.max() + 10.0 * (fin.max() - fin.min()))
        c_tr = np.where(np.isposinf(c_tr), cap, c_tr)
        c_te = np.where(np.isposinf(c_te), cap, c_te)
    return c_tr, c_te, n_inf


def parse_range(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--dataset", choices=["cifar10", "cifar100"], required=True)
    ap.add_argument("--model", default="resnet50")
    ap.add_argument("--epochs", required=True)
    ap.add_argument("--seeds", default="42-91")
    ap.add_argument("--check_seed", type=int, default=42)
    ap.add_argument("--backend", choices=["torch", "numpy"], default="torch")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry_run", action="store_true", help="random features instead of checkpoints (pipeline test)")
    args = ap.parse_args()
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tag, seeds, epochs = TAG[args.dataset], parse_range(args.seeds), parse_epochs(args.epochs)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    done = set()
    if os.path.exists(args.out) and os.path.getsize(args.out) > 0:
        with open(args.out) as f:
            done = {(int(r["epoch"]), int(r["seed"])) for r in csv.DictReader(f)}
    new = not os.path.exists(args.out) or os.path.getsize(args.out) == 0
    fout = open(args.out, "a", newline=""); w = csv.writer(fout)
    if new:
        w.writerow(["tag", "epoch", "seed", "dc", "mmd_c", "p", "vrt_p", "n_inf"]); fout.flush()
    print(f"[{tag}] MMD on c-hat, epochs {epochs[0]} to {epochs[-1]} ({len(epochs)}), seeds {seeds[0]}-{seeds[-1]}, "
          f"check seed {args.check_seed}, device {device}", flush=True)
    model = None if args.dry_run else SupConResNet(name=args.model).to(device)
    for ep in epochs:
        todo = [s for s in seeds if (ep, s) not in done]
        if not todo:
            continue
        path = os.path.join(args.ckpt_dir, f"ckpt_epoch_{ep}.pth")
        if not args.dry_run and not os.path.exists(path):
            print(f"  ep{ep:4d}  no checkpoint, skip", flush=True); continue
        t0 = time.time()
        if args.dry_run:
            rng0 = np.random.default_rng(ep)
            h_tr = rng0.standard_normal((12000, 64)).astype(np.float32)
            h_te = (rng0.standard_normal((10000, 64)) + 0.01 * ep).astype(np.float32)
        else:
            load_ckpt(model, path, device)
            feats = extract(model, args.dataset, device)
            h_tr, h_te = feats["h_tr"], feats["h_te"]
        e_tr, e_te = l2n(h_tr).astype(np.float32), l2n(h_te).astype(np.float32)   # as ssl_s50_fill.py for vrt b10000
        D = e_tr.shape[1]
        t_feat = time.time() - t0
        ps = []
        for s in todo:
            rng = np.random.RandomState(s)                                            # the draw of vrt_test(seed = s)
            perm = rng.permutation(len(e_tr))
            ref = e_tr[perm[:M]]
            q_tr = e_tr[perm[M:M + min(N_QTR, len(e_tr) - M)]]
            c_tr = compute_chat(q_tr, ref, D, K)
            c_te = compute_chat(e_te, ref, D, K)
            dc = float(np.median(c_te) - np.median(c_tr))
            c_tr, c_te, n_inf = cap_infinite(c_tr, c_te)                              # see cap_infinite
            if float(np.median(c_te) - np.median(c_tr)) != dc:
                raise RuntimeError(f"ep{ep} seed {s}: capping changed the VRT median difference")
            stat, p = mmd_test_fast(c_tr[:, None], c_te[:, None], n_sub=10000, n_perm=N_PERM, seed=s, backend=args.backend)
            if not np.isfinite(stat):
                raise RuntimeError(f"ep{ep} seed {s}: MMD statistic {stat} is not finite")
            vp = ""
            if s == args.check_seed:
                _, vp = vrt_test(e_tr, e_te, D, k=K, M=M, n_qtr=N_QTR, n_perm=N_PERM, seed=s)
                vp = f"{vp:.6f}"
            w.writerow([tag, ep, s, f"{dc:+.6f}", f"{stat:.8e}", f"{p:.6f}", vp, n_inf]); ps.append(p)
        fout.flush()
        print(f"  ep{ep:4d}  median p {np.median(ps):.4f}  {sum(p < 0.05 for p in ps)}/{len(ps)} seeds < 0.05  "
              f"(features {t_feat:.0f} s, total {time.time() - t0:.0f} s)", flush=True)
    fout.close(); print("done", flush=True)


if __name__ == "__main__":
    main()
