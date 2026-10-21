"""SUPERSEDED (camera-ready code release note). This script belongs to the 10-seed stage of the project and resumes
by counting rows per epoch, which duplicates seeds when an epoch was partly written. The 50-seed results of the
paper come from ssl_mmdagg5k_seeds.py, which resumes by (epoch, seed). Kept for provenance; do not use for new runs."""
"""
ssl_mmdagg5k_fill.py — MMDAgg at 5,000 points per split on the SimCLR models (matched tier), features
extracted on the fly as in ssl_baselines_fill.py, 10 seeds, on an epoch grid chosen with --every.

Coarse-to-fine use: first --every 10 over 1 to 1,000 to locate the onset region, then --every 1 on a
window around it, then --every 5 after it to confirm sustained rejection. Every run appends to its own
file sslagg_<dataset>_<lo>-<hi>_e<every>.csv (model,epoch,budget,stat,seed,acc,p; p = 0/1 reject flag),
all of which harvest_ssl.py reads together (stat mmdagg, budget b5000). Resume: epochs with 10 rows skipped.
Runs on CPU if no GPU is available (extraction is then about 10 minutes per checkpoint, the ten MMDAgg
tests about 50 minutes).
"""
import argparse, os, time, glob
import numpy as np
import torch

from supcon_fill_vrt import SupConResNet, load_ckpt, extract, l2n
from baselines2_stats import mmdagg_test, mmdagg_fn

OUT_DIR = "/home3/p318770/venvs/overfitting/benchmark_final/ssl_baselines"
SEEDS = list(range(42, 52))
BUDGET = 5000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--dataset", choices=["cifar10", "cifar100"], required=True)
    ap.add_argument("--model", default="resnet50")
    ap.add_argument("--lo", type=int, default=1)
    ap.add_argument("--hi", type=int, default=1000)
    ap.add_argument("--every", type=int, default=10, help="evaluate epochs divisible by this step (1 = every epoch)")
    ap.add_argument("--out_dir", default=OUT_DIR)
    args = ap.parse_args()
    if mmdagg_fn is None:
        raise SystemExit("mmdagg not importable: load the module first, then activate the venv")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, f"sslagg_{args.dataset}_{args.lo}-{args.hi}_e{args.every}.csv")

    # epochs already complete in ANY sslagg file of this dataset (coarse and fine runs share the grid)
    done = {}
    for f in glob.glob(os.path.join(args.out_dir, f"sslagg_{args.dataset}_*.csv")):
        with open(f) as fh:
            next(fh, None)
            for line in fh:
                ep0 = int(line.split(",")[1]); done[ep0] = done.get(ep0, 0) + 1
    done_eps = {e for e, c in done.items() if c >= len(SEEDS)}

    names = [f for f in os.listdir(args.ckpt_dir) if f.startswith("ckpt_epoch_") and f.endswith(".pth")]
    names.sort(key=lambda s: int(s[len("ckpt_epoch_"):-len(".pth")]))
    sel = []
    for n in names:
        ep = int(n[len("ckpt_epoch_"):-len(".pth")])
        if args.lo <= ep <= args.hi and (args.every <= 1 or ep % args.every == 0) and ep not in done_eps:
            sel.append((ep, n))
    print(f"{args.dataset}: {len(sel)} checkpoints to do in [{args.lo}, {args.hi}] step {args.every} "
          f"({len(done_eps)} already complete), device {device}", flush=True)

    new = not os.path.exists(out_csv)
    fout = open(out_csv, "a")
    if new:
        fout.write("model,epoch,budget,stat,seed,acc,p\n")
    model = SupConResNet(name=args.model).to(device)
    for ep, name in sel:
        t0 = time.time()
        load_ckpt(model, os.path.join(args.ckpt_dir, name), device)
        feats = extract(model, args.dataset, device)
        e_tr, e_te = l2n(feats["h_tr"]).astype(np.float32), l2n(feats["h_te"]).astype(np.float32)
        t_feat = time.time() - t0
        rej = []
        for seed in SEEDS:
            r = mmdagg_test(e_tr, e_te, n_sub=BUDGET, seed=seed); rej.append(r)
            fout.write(f"{args.dataset},{ep},b{BUDGET},mmdagg,{seed},nan,{float(r):.1f}\n")
        fout.flush()
        print(f"ep {ep:4d}  mmdagg{BUDGET} {sum(rej)}/{len(rej)} reject  (feat {t_feat:.0f} s, total {time.time() - t0:.0f} s)", flush=True)
    fout.close()
    print(f"done -> {out_csv}")


if __name__ == "__main__":
    main()
