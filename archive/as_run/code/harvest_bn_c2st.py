"""harvest_bn_c2st.py — C2ST on ResNet-50 (CIFAR-100) under eval-mode and train-mode BatchNorm extraction, 10,000 points
per split, 50 seeds, the decision rule of Table 1 (median p < 0.05 and median held-out accuracy > 0.55 at a checkpoint;
onset = first evaluated checkpoint from which every later evaluated checkpoint rejects).
  eval mode  : the s50 results, s50/c2st/c2st_r50_b*.csv (model r50, budget b10000)
  train mode : c2st_bntrain_r50_tasks.sh, s50/bntrain_c2st/c2st_r50_bntrain_b10000_c*.csv
Prints the two onsets and a per-epoch table (median accuracy, median p, seeds) over a window, default 100 to 200.
Usage:  python harvest_bn_c2st.py [lo hi]
"""
import sys, glob, csv, collections
import numpy as np

S50 = "/home3/p318770/venvs/overfitting/benchmark_final/s50"
lo, hi = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (100, 200)


def load(pattern, model):
    d = collections.defaultdict(dict)                     # epoch -> {seed: (p, acc)}
    for f in glob.glob(pattern):
        for r in csv.DictReader(open(f)):
            if r["model"] == model and r["budget"] == "b10000":
                d[int(r["epoch"])].setdefault(int(r["seed"]), (float(r["p"]), float(r["acc"])))
    return d


def decide(v):
    ps = np.array([p for p, _ in v.values()]); accs = np.array([a for _, a in v.values()])
    return np.median(ps) < 0.05 and np.median(accs) > 0.55, np.median(ps), np.median(accs)


runs = {"eval": load(f"{S50}/c2st/c2st_r50_b*.csv", "r50"),
        "train": load(f"{S50}/bntrain_c2st/c2st_r50_bntrain_b*.csv", "r50_bntrain")}
dec = {}
for mode, d in runs.items():
    if not d:
        print(f"{mode:5s}: no results yet"); continue
    eps = sorted(d); dec[mode] = {e: decide(d[e]) for e in eps}
    rej = {e: dec[mode][e][0] for e in eps}
    onset = next((e for i, e in enumerate(eps) if all(rej[x] for x in eps[i:])), None)
    first = next((e for e in eps if rej[e]), None)
    nseed = sorted({len(v) for v in d.values()})
    missing = sorted(set(range(eps[0], eps[-1] + 1)) - set(eps))
    print(f"{mode:5s}: epochs {eps[0]}-{eps[-1]} ({len(eps)} present, missing {missing[:12]}{' ...' if len(missing) > 12 else ''}), "
          f"seeds per epoch {nseed[0]}{'' if len(nseed) == 1 else f' to {nseed[-1]}'}   first reject {first}   "
          f"sustained onset {onset if onset is not None else '-'}")
print(f"\nper epoch {lo}-{hi}: median held-out accuracy / median p (R = rejects under the rule)")
print(f"{'epoch':>5s} {'eval mode':>20s} {'train mode':>20s}")
for e in range(lo, hi + 1):
    cells = []
    for mode in ("eval", "train"):
        x = dec.get(mode, {}).get(e)
        cells.append(f"{'.':>20s}" if x is None else f"{x[2]:.4f} / {x[1]:.4f} {'R' if x[0] else ' '}".rjust(20))
    if any(c.strip() != "." for c in cells):
        print(f"{e:5d} " + " ".join(cells))
print("\nrule: median p < 0.05 and median held-out accuracy > 0.55 (n_te = 10,000); eval-mode onset in Table 1: 121")
