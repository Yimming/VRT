"""
vrt_forward_budget.py — forward VRT (TRAIN split as reference, the paper's direction) on precomputed
per-epoch npz features, at the two unified budgets.

  large : 10,000 per split.  M = 2,000 train reference, 8,000 remaining train queries vs ALL test points
          (10,000), k = 60, 2,000 permutations   (paper protocol, Table 1). Function vrt_low_lib.vrt_full,
          the verbatim body of supcon_fill_vrt.vrt_test, so the random stream is the one of the published
          SimCLR per-epoch numbers.
  small : 1,500 per split.   M = 500 train reference, 1,000 train queries vs 1,500 test queries,
          k = 15 (k/M = 0.03 preserved), 2,000 permutations   (rebuttal small-budget protocol). Function
          vrt_low_lib.vrt_low, verbatim from ksweep15_fill.py.

Scoring (log cap volumes, Renyi spacings) is the verbatim code of the supplementary phase_multiseed_vrt.py.
Features are L2-normalized in float64 as in ksweep15_fill.py. Median difference, two-sided,
p = (count + 1) / (n_perm + 1). The default --seeds 42-51 is the 10-seed default of the earlier stage; the
published runs used --seeds 42-91 (50 seeds).

  python vrt_forward_budget.py --tag imgnet_simple --budget large \
      --npz_fmt "/scratch/p318770/model_zoos/ladder_feats/imgnet_simple/ep{ep:03d}.npz" \
      --epochs 1,5,20,50,100,150,200,250,300 --out $BF/vrt_fwd/vrt_fwd_imgnet_simple_large.csv

CSV: tag,budget,epoch,seed,dc,p   (dc = median c_te minus median c_tr). Resume: (epoch, seed) rows present are skipped.
"""
import os, argparse, csv, time
import numpy as np
from vrt_low_lib import vrt_low, vrt_full, l2n
from release_utils import validate_features

BUDGETS = {  # name: (M reference, k, n train queries, n test queries)
    "large": (2000, 60, 8000, 10000),
    "small": (500, 15, 1000, 1500),
}
KEY_PAIRS = [("h_tr", "h_te"), ("X_tr", "X_te"), ("tr", "te"), ("train", "test"), ("emb_tr", "emb_te")]


def load_epoch(fmt, ep):
    f = fmt.format(ep=ep)
    if not os.path.exists(f):
        return None
    d = np.load(f)
    for a, b in KEY_PAIRS:
        if a in d.files and b in d.files:
            return d[a], d[b]
    raise KeyError(f"unknown keys in {f}: {d.files}")


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
    ap.add_argument("--tag", required=True)
    ap.add_argument("--budget", choices=list(BUDGETS), required=True)
    ap.add_argument("--npz_fmt", required=True, help="format string with {ep}, one npz per epoch holding both splits")
    ap.add_argument("--epochs", required=True)
    ap.add_argument("--seeds", default="42-51")
    ap.add_argument("--n_perm", type=int, default=2000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    M, k, n_qtr, n_qte = BUDGETS[args.budget]
    seeds, epochs = parse_range(args.seeds), parse_range(args.epochs)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            for r in csv.DictReader(f):
                done.add((int(r["epoch"]), int(r["seed"])))
    new = not os.path.exists(args.out) or os.path.getsize(args.out) == 0   # a task killed before its first flush leaves an empty file
    fout = open(args.out, "a", newline=""); w = csv.writer(fout)
    if new:
        w.writerow(["tag", "budget", "epoch", "seed", "dc", "p"]); fout.flush()
    print(f"[{args.tag}] forward VRT, budget {args.budget}: M={M}, k={k}, {n_qtr} train vs {n_qte} test queries, "
          f"{args.n_perm} permutations, seeds {seeds[0]}-{seeds[-1]}", flush=True)

    n_missing = 0
    for ep in epochs:
        todo = [s for s in seeds if (ep, s) not in done]
        if not todo:
            continue
        loaded = load_epoch(args.npz_fmt, ep)
        if loaded is None:
            n_missing += 1; continue     # epoch ranges may cover epochs with no saved features
        t0 = time.time()
        validate_features(*loaded, names=("train features", "test features"))
        e_tr, e_te = l2n(loaded[0].astype(np.float64)), l2n(loaded[1].astype(np.float64))
        D = e_tr.shape[1]
        if len(e_tr) < M + n_qtr or len(e_te) < n_qte:
            print(f"  ep{ep:4d}  only {len(e_tr)} train / {len(e_te)} test points, budget needs {M + n_qtr} / {n_qte}; skip", flush=True)
            continue
        if args.budget == "large" and len(e_te) != n_qte:
            print(f"  ep{ep:4d}  large budget uses all {len(e_te)} test points (protocol: 10,000)", flush=True)
        ps = []
        for s in todo:
            if args.budget == "large":
                dc, p = vrt_full(e_tr, e_te, D, k=k, M=M, n_qtr=n_qtr, n_perm=args.n_perm, seed=s)
            else:
                dc, p = vrt_low(e_tr, e_te, D, k=k, M=M, n_qtr=n_qtr, n_qte=n_qte, n_perm=args.n_perm, seed=s)
            w.writerow([args.tag, args.budget, ep, s, f"{dc:+.6f}", f"{p:.6f}"]); ps.append(p)
        fout.flush()
        print(f"  ep{ep:4d}  median p {np.median(ps):.4f}  {sum(p < 0.05 for p in ps)}/{len(ps)} seeds < 0.05  ({time.time() - t0:.0f} s)", flush=True)
    fout.close(); print(f"done ({n_missing} epochs in the range had no feature file)")


if __name__ == "__main__":
    main()
