"""SUPERSEDED (camera-ready code release note). This script belongs to the 10-seed stage of the project and resumes
by counting rows per epoch, which duplicates seeds when an epoch was partly written. The 50-seed results of the
paper come from ssl_s50_fill.py (--group base / knn / rev), which resumes by (epoch, seed). Kept for provenance; do not use for new runs."""
"""
ssl_baselines_fill.py — per-epoch baselines for the SimCLR models (camera-ready, unified protocol).

Mirrors supcon_fill_vrt.py: for every checkpoint in [lo, hi] the backbone features of the train and test
split are extracted on the fly (nothing is saved to disk), l2-normalized, and the paper baselines are run
with 10 seeds (42 to 51):
  mmd     1,500 and 10,000 per split, 2,000 permutations   (mmd_tst_fast.mmd_test_fast, == methods_v2.mmd_test)
  ed      1,500 and 10,000 per split, 2,000 permutations   (energy_tst_fast.energy_test_fast, == methods_v2.energy_distance_test)
  c2st    1,500 and 8,000 (Table 1 budget) per split every epoch, 10,000 per split every 5th epoch
          (--big_every, the rebuttal's big-tier protocol), normal-approximation p and held-out accuracy
  mmdagg  1,500 per split, B1 = B2 = 2,000 wild-bootstrap, binary reject
Output CSV (same layout as baselines2_stats.py): model,epoch,budget,stat,seed,acc,p
Resume: an epoch is skipped when it already has the full set of rows.

Run from /scratch/p318770/ssl_supcon_code (imports train_simclr_supcon via supcon_fill_vrt):
  python ssl_baselines_fill.py --ckpt_dir <SimCLR dir> --dataset cifar10 --lo 1 --hi 100
"""
import argparse, os, time
import numpy as np
import torch

from supcon_fill_vrt import SupConResNet, load_ckpt, extract, l2n
from mmd_tst_fast import mmd_test_fast
from energy_tst_fast import energy_test_fast
from baselines2_stats import c2st_test, mmdagg_test, mmdagg_fn

OUT_DIR = "/home3/p318770/venvs/overfitting/benchmark_final/ssl_baselines"
SEEDS = list(range(42, 52))
MMD_BUDGETS = (1500, 10000)
ED_BUDGETS = (1500, 10000)
C2ST_BUDGETS = (1500, 8000, 10000)
MMDAGG_BUDGETS = (1500,)


def c2st_budgets_for(ep, args):
    """1,500 and 8,000 every epoch; 10,000 only every --big_every epochs (0 = every epoch)."""
    big = args.big_every <= 1 or ep % args.big_every == 0
    return tuple(b for b in C2ST_BUDGETS if b < 10000 or big)


def expected_rows(ep, args):
    n = len(SEEDS)
    per_epoch = n * (len(MMD_BUDGETS) + len(ED_BUDGETS))
    if not args.skip_c2st:
        per_epoch += n * len(c2st_budgets_for(ep, args))
    if not args.skip_mmdagg and mmdagg_fn is not None:
        per_epoch += n * len(MMDAGG_BUDGETS)
    return per_epoch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--dataset", choices=["cifar10", "cifar100"], required=True)
    ap.add_argument("--model", default="resnet50")
    ap.add_argument("--lo", type=int, default=0)
    ap.add_argument("--hi", type=int, default=10**9)
    ap.add_argument("--n_perm", type=int, default=2000)
    ap.add_argument("--big_every", type=int, default=5, help="C2ST 10,000 tier every N epochs (1 = every epoch)")
    ap.add_argument("--skip_c2st", action="store_true")
    ap.add_argument("--skip_mmdagg", action="store_true")
    ap.add_argument("--out_dir", default=OUT_DIR)
    ap.add_argument("--dry_run", action="store_true", help="random features instead of checkpoints (pipeline test)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    backend = "torch"
    tag = f"{args.dataset}_{os.path.basename(os.path.normpath(args.ckpt_dir))}"
    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, f"sslbase_{args.dataset}_{args.lo}-{args.hi}.csv")

    if args.dry_run:
        names = [f"ckpt_epoch_{e}.pth" for e in range(args.lo, args.hi + 1)]
    else:
        names = [f for f in os.listdir(args.ckpt_dir) if f.startswith("ckpt_epoch_") and f.endswith(".pth")]
        names.sort(key=lambda s: int(s[len("ckpt_epoch_"):-len(".pth")]))
        names = [n for n in names if args.lo <= int(n[len("ckpt_epoch_"):-len(".pth")]) <= args.hi]
    print(f"{tag}: {len(names)} checkpoints in [{args.lo}, {args.hi}], mmdagg available: {mmdagg_fn is not None}", flush=True)

    done = {}
    if os.path.exists(out_csv):
        with open(out_csv) as f:
            next(f, None)
            for line in f:
                ep0 = int(line.split(",")[1]); done[ep0] = done.get(ep0, 0) + 1
    done_eps = {e for e, c in done.items() if c >= expected_rows(e, args)}
    print(f"resume: {len(done_eps)} epochs complete, skipping", flush=True)
    new = not os.path.exists(out_csv)
    fout = open(out_csv, "a")
    if new:
        fout.write("model,epoch,budget,stat,seed,acc,p\n")

    def emit(ep, budget, stat, seed, acc, p):
        fout.write(f"{args.dataset},{ep},b{budget},{stat},{seed},{acc:.6f},{p:.6f}\n")

    model = None if args.dry_run else SupConResNet(name=args.model).to(device)
    for name in names:
        ep = int(name[len("ckpt_epoch_"):-len(".pth")])
        if ep in done_eps:
            continue
        t0 = time.time()
        if args.dry_run:
            rng = np.random.default_rng(ep)
            h_tr = rng.standard_normal((50000, 2048)).astype(np.float32)
            h_te = (rng.standard_normal((10000, 2048)) + 0.002 * ep).astype(np.float32)
        else:
            load_ckpt(model, os.path.join(args.ckpt_dir, name), device)
            feats = extract(model, args.dataset, device)
            h_tr, h_te = feats["h_tr"], feats["h_te"]
        e_tr, e_te = l2n(h_tr).astype(np.float32), l2n(h_te).astype(np.float32)
        t_feat = time.time() - t0

        med = {}
        for budget in MMD_BUDGETS:
            ps = []
            for seed in SEEDS:
                _, p = mmd_test_fast(e_tr, e_te, budget, args.n_perm, seed, backend); ps.append(p)
                emit(ep, budget, "mmd", seed, np.nan, p)
            med[f"mmd{budget}"] = np.median(ps)
        for budget in ED_BUDGETS:
            ps = []
            for seed in SEEDS:
                _, p = energy_test_fast(e_tr, e_te, budget, args.n_perm, seed, backend); ps.append(p)
                emit(ep, budget, "ed", seed, np.nan, p)
            med[f"ed{budget}"] = np.median(ps)
        if not args.skip_c2st:
            for budget in c2st_budgets_for(ep, args):
                ps, accs = [], []
                for seed in SEEDS:
                    acc, p = c2st_test(e_tr, e_te, n_sub=budget, seed=seed); ps.append(p); accs.append(acc)
                    emit(ep, budget, "c2st", seed, acc, p)
                med[f"c2st{budget}"] = f"{np.median(ps):.3f}/acc{np.median(accs):.3f}"
        if not args.skip_mmdagg and mmdagg_fn is not None:
            for budget in MMDAGG_BUDGETS:
                rej = []
                for seed in SEEDS:
                    r = mmdagg_test(e_tr, e_te, n_sub=budget, seed=seed); rej.append(r)
                    emit(ep, budget, "mmdagg", seed, np.nan, float(r))
                med[f"mmdagg{budget}"] = f"{sum(rej)}/{len(rej)} reject"
        fout.flush()
        print(f"ep {ep:4d}  " + "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in med.items())
              + f"   (feat {t_feat:.0f} s, total {time.time() - t0:.0f} s)", flush=True)

    fout.close()
    print(f"done -> {out_csv}")


if __name__ == "__main__":
    main()
