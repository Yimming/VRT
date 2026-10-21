"""LEGACY (camera-ready code release note): onsets of the 10-seed SimCLR baseline files (earlier stage); referenced by historical task files; the published SimCLR onsets come from the ssl/ and ssl_baselines/ 50-seed files. Kept as run for provenance; not an entry point of the release."""
"""harvest_ssl.py — onsets of the per-epoch SimCLR baselines (sslbase_<dataset>_<lo>-<hi>.csv chunks).
Usage: python harvest_ssl.py [dir]   (default /home3/p318770/venvs/overfitting/benchmark_final/ssl_baselines)
Decision per epoch, over the 10 seeds: mmd / ed / c2st median p < 0.05 (c2st additionally median held-out
accuracy > 0.55, the Table 1 convention); mmdagg majority of seeds rejecting (>= 6 of 10).
Onset = first evaluated epoch from which every later evaluated epoch rejects. VRT per-epoch onsets from the
rebuttal harvest: CIFAR-10 189, CIFAR-100 238 (paper protocol, 10 seeds, 2,000 permutations).
"""
import sys, glob, os, csv, collections
import numpy as np

root = sys.argv[1] if len(sys.argv) > 1 else "/home3/p318770/venvs/overfitting/benchmark_final/ssl_baselines"
VRT = {"cifar10": 189, "cifar100": 238}

rows = collections.defaultdict(lambda: collections.defaultdict(dict))   # (ds, stat, budget) -> epoch -> seed -> (acc, p)
for f in glob.glob(os.path.join(root, "sslbase_*.csv")) + glob.glob(os.path.join(root, "sslvrt_*.csv")) + glob.glob(os.path.join(root, "sslagg_*.csv")):
    with open(f) as fh:
        for r in csv.DictReader(fh):
            key = (r["model"], r["stat"], r["budget"])
            rows[key][int(r["epoch"])][int(r["seed"])] = (float(r["acc"]), float(r["p"]))

def decide(stat, vals):
    accs = np.array([a for a, _ in vals]); ps = np.array([p for _, p in vals])
    if stat == "mmdagg":
        return np.mean(ps) >= 0.6                      # p column holds the 0/1 reject flag
    if stat == "c2st":
        return np.median(ps) < 0.05 and np.median(accs) > 0.55
    return np.median(ps) < 0.05

def onset(eps, rej):
    for i, e in enumerate(eps):
        if all(rej[x] for x in eps[i:]):
            return e
    return None

ORDER = [("vrt_low", "b1500"), ("mmd", "b1500"), ("mmdagg", "b1500"), ("c2st", "b8000"), ("ed", "b1500"),
         ("mmd", "b10000"), ("ed", "b10000"), ("mmdagg", "b5000"), ("c2st", "b1500"), ("c2st", "b10000")]
for ds in ("cifar10", "cifar100"):
    print(f"=== SimCLR {ds}: VRT per-epoch onset {VRT[ds]} ===")
    print(f"{'stat':7s} {'budget':7s} {'epochs':>6s} {'range':>10s} {'first':>6s} {'onset':>6s} {'seeds':>6s}")
    for stat, budget in ORDER:
        d = rows.get((ds, stat, budget))
        if not d:
            print(f"{stat:7s} {budget:7s} {'--':>6s} (no rows)"); continue
        eps = sorted(d)
        rej = {e: decide(stat, list(d[e].values())) for e in eps}
        first = next((e for e in eps if rej[e]), None)
        nseeds = min(len(d[e]) for e in eps)
        f = lambda x: "--" if x is None else str(x)
        print(f"{stat:7s} {budget:7s} {len(eps):6d} {f'{eps[0]}-{eps[-1]}':>10s} {f(first):>6s} {f(onset(eps, rej)):>6s} {nseeds:6d}")
    # gaps in coverage (chunks not finished yet)
    covered = sorted({e for (m, s, b), d in rows.items() if m == ds for e in d})
    if covered:
        missing = sorted(set(range(covered[0], covered[-1] + 1)) - set(covered))
        print(f"coverage {covered[0]}-{covered[-1]}, {len(missing)} epochs missing inside" + (f" (first {missing[:8]})" if missing else ""))
    print()
