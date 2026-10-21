"""SUPERSEDED (camera-ready code release note). This script belongs to the 10-seed stage of the project and resumes
by counting rows per epoch, which duplicates seeds when an epoch was partly written. The 50-seed results of the
paper come from ssl_s50_fill.py (--group vrt), which resumes by (epoch, seed). Kept for provenance; do not use for new runs."""
"""
ssl_vrt_low_fill.py — small-budget VRT (rebuttal protocol, k = 15) on the SimCLR models, every epoch,
features extracted on the fly exactly as in supcon_fill_vrt.py / ssl_baselines_fill.py.

Protocol (vrt_low_lib.vrt_low, verbatim from ksweep15_fill.py): 1,500 training samples per test, of which
M = 500 form the reference and 1,000 are train queries; 1,500 test queries; k = 15; 2,000 permutations;
two-sided median test; seeds 42 to 51 with RandomState(seed).

Output: <out_dir>/sslvrt_<dataset>_<lo>-<hi>.csv with model,epoch,budget,stat,seed,acc,p
(budget b1500, stat vrt_low, acc = the observed median difference T), picked up by harvest_ssl.py.
Resume: epochs with all 10 seeds present are skipped.
"""
import argparse, os, time
import numpy as np
import torch

from supcon_fill_vrt import SupConResNet, load_ckpt, extract, l2n
from vrt_low_lib import vrt_low

OUT_DIR = "/home3/p318770/venvs/overfitting/benchmark_final/ssl_baselines"
SEEDS = list(range(42, 52))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--dataset", choices=["cifar10", "cifar100"], required=True)
    ap.add_argument("--model", default="resnet50")
    ap.add_argument("--lo", type=int, default=0)
    ap.add_argument("--hi", type=int, default=10**9)
    ap.add_argument("--out_dir", default=OUT_DIR)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, f"sslvrt_{args.dataset}_{args.lo}-{args.hi}.csv")

    if args.dry_run:
        names = [f"ckpt_epoch_{e}.pth" for e in range(args.lo, args.hi + 1)]
    else:
        names = [f for f in os.listdir(args.ckpt_dir) if f.startswith("ckpt_epoch_") and f.endswith(".pth")]
        names.sort(key=lambda s: int(s[len("ckpt_epoch_"):-len(".pth")]))
        names = [n for n in names if args.lo <= int(n[len("ckpt_epoch_"):-len(".pth")]) <= args.hi]
    print(f"{args.dataset}: {len(names)} checkpoints in [{args.lo}, {args.hi}]", flush=True)

    done = {}
    if os.path.exists(out_csv):
        with open(out_csv) as f:
            next(f, None)
            for line in f:
                ep0 = int(line.split(",")[1]); done[ep0] = done.get(ep0, 0) + 1
    done_eps = {e for e, c in done.items() if c >= len(SEEDS)}
    new = not os.path.exists(out_csv)
    fout = open(out_csv, "a")
    if new:
        fout.write("model,epoch,budget,stat,seed,acc,p\n")
    print(f"resume: {len(done_eps)} epochs complete, skipping", flush=True)

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
        e_tr, e_te = l2n(h_tr.astype(np.float64)), l2n(h_te.astype(np.float64))   # float64 as in ksweep15_fill.py
        D = e_tr.shape[1]
        ps = []
        for seed in SEEDS:
            T, p = vrt_low(e_tr, e_te, D, seed=seed)
            fout.write(f"{args.dataset},{ep},b1500,vrt_low,{seed},{T:.6e},{p:.6f}\n"); ps.append(p)
        fout.flush()
        print(f"ep {ep:4d}  vrt_low med_p={np.median(ps):.4f} reject {sum(p < 0.05 for p in ps)}/10  ({time.time() - t0:.0f} s)", flush=True)
    fout.close()
    print(f"done -> {out_csv}")


if __name__ == "__main__":
    main()
