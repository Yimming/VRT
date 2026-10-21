"""ssl_peek.py — read the SimCLR 50-seed results of one dataset while the run is still going.
For every (statistic, budget) in the fast/slow CSVs: epochs present, missing epochs below the last one present
(chunks still running), first rejecting epoch, sustained onset over the epochs present, and the epochs after the
first rejection that do not reject. Then a per-epoch window (median p, seeds < 0.05) for the 10,000 tier.
  python ssl_peek.py cifar10 [lo hi]        (window default 170-300)
"""
import sys, csv, glob, collections
import numpy as np

BF = "/home3/p318770/venvs/overfitting/benchmark_final/s50/ssl"
ds = sys.argv[1] if len(sys.argv) > 1 else "cifar10"
lo, hi = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (170, 300)
d = collections.defaultdict(dict)
for f in glob.glob(f"{BF}/ssl_{ds}_*_c*.csv"):
    for r in csv.DictReader(open(f)):
        d[(r["stat"], r["budget"])].setdefault(int(r["epoch"]), {})[int(r["seed"])] = (float(r["p"]), float(r["acc"]))


def rej(stat, v):
    ps = np.array([p for p, _ in v.values()])
    if stat == "mmdagg":
        return ps.mean() >= 0.6
    if stat == "c2st":
        return np.median(ps) < 0.05 and np.median([a for _, a in v.values()]) > 0.55
    return np.median(ps) < 0.05


R = {}
for key in sorted(d):
    stat, b = key; eps = sorted(d[key]); R[key] = {e: rej(stat, d[key][e]) for e in eps}
    first = next((e for e in eps if R[key][e]), None)
    onset = next((e for i, e in enumerate(eps) if all(R[key][x] for x in eps[i:])), None)
    nseed = min(len(v) for v in d[key].values())
    missing = sorted(set(range(eps[0], eps[-1] + 1)) - set(eps))
    print(f"{stat:8s} {b:7s} epochs {eps[0]}-{eps[-1]}, {len(eps)} present, {len(missing)} missing below {eps[-1]}, "
          f"min seeds {nseed}   first reject {first}   sustained onset {onset}")
    if first is not None:
        fails = [e for e in eps if e > first and not R[key][e]]
        if fails:
            print(f"         not rejecting after the first rejection: {fails[:40]}{' ...' if len(fails) > 40 else ''}")
print()
cols = [k for k in sorted(d) if k[1] == "b10000"]
print(f"per-epoch window {lo}-{hi}, 10,000 tier: median p and seeds < 0.05")
print(f"{'epoch':>5s} " + " ".join(f"{s:>14s}" for s, _ in cols))
for e in range(lo, hi + 1):
    if not any(e in d[k] for k in cols):
        continue
    cells = []
    for k in cols:
        v = d[k].get(e)
        if v is None:
            cells.append(f"{'.':>14s}"); continue
        ps = np.array([p for p, _ in v.values()])
        cells.append(f"{np.median(ps):6.3f} {int((ps < 0.05).sum()):3d}/{len(ps):<3d}")
    print(f"{e:5d} " + " ".join(cells))
