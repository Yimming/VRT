"""LEGACY (camera-ready code release note): seed-robustness comparison between the 10-seed and the 50-seed SimCLR onsets (transition stage). Kept as run for provenance; not an entry point of the release."""
"""harvest_ssl_seeds.py — seed-robustness of the SimCLR large-budget onsets. Merges the published 10-seed
per-epoch results (VRT: vrt_<dataset>_SimCLR*.csv, backbone space; MMD/ED: ssl_baselines/sslbase_*.csv at
b10000) with the extra seeds of ssl_seeds_fill.py (ssl_baselines/sslseeds_*.csv), and reports for each method
the first rejection and the sustained onset (a) with the 10 published seeds and (b) with all seeds available
per epoch (50 inside the extra-seed window, 10 outside). Lists the near-threshold epochs inside the window.
Usage: python harvest_ssl_seeds.py [benchmark_final dir]
"""
import sys, glob, os, csv, collections
import numpy as np

BF = sys.argv[1] if len(sys.argv) > 1 else "/home3/p318770/venvs/overfitting/benchmark_final"
PUB = set(range(42, 52))


def onset(eps, med):
    for i, e in enumerate(eps):
        if all(med[x] < 0.05 for x in eps[i:]):
            return e
    return None


fmt = lambda x: "--" if x is None else str(x)

for ds in ("cifar10", "cifar100"):
    P = {"vrt": collections.defaultdict(dict), "mmd": collections.defaultdict(dict), "ed": collections.defaultdict(dict)}
    for f in glob.glob(f"{BF}/vrt_{ds}_SimCLR*.csv"):
        with open(f) as fh:
            for r in csv.DictReader(fh):
                if r.get("space", "backbone") == "backbone":
                    P["vrt"][int(r["epoch"])][int(r["seed"])] = float(r["p"])
    for f in glob.glob(f"{BF}/ssl_baselines/sslbase_{ds}_*.csv") + glob.glob(f"{BF}/ssl_baselines/sslseeds_{ds}_*.csv"):
        with open(f) as fh:
            for r in csv.DictReader(fh):
                if r["budget"] == "b10000" and r["stat"] in P:
                    P[r["stat"]][int(r["epoch"])][int(r["seed"])] = float(r["p"])
    extra = sorted({e for st in P for e in P[st] if any(s not in PUB for s in P[st][e])})
    print(f"=== SimCLR {ds}, 10,000 per split: published 10 seeds vs all seeds "
          f"(extra seeds on {len(extra)} epochs{f', {extra[0]}-{extra[-1]}' if extra else ''}) ===")
    print(f"{'method':6s} {'first(10)':>9s} {'onset(10)':>9s} {'first(all)':>10s} {'onset(all)':>10s} {'max seeds':>9s}")
    near = collections.defaultdict(dict)
    for st in ("vrt", "mmd", "ed"):
        d = P[st]
        if not d:
            print(f"{st:6s} (no rows)"); continue
        eps = sorted(d)
        med10 = {e: float(np.median([p for s, p in d[e].items() if s in PUB])) for e in eps if any(s in PUB for s in d[e])}
        medall = {e: float(np.median(list(d[e].values()))) for e in eps}
        eps10 = sorted(med10)
        f10 = next((e for e in eps10 if med10[e] < 0.05), None); fall = next((e for e in eps if medall[e] < 0.05), None)
        print(f"{st:6s} {fmt(f10):>9s} {fmt(onset(eps10, med10)):>9s} {fmt(fall):>10s} {fmt(onset(eps, medall)):>10s} {max(len(v) for v in d.values()):9d}")
        for e in extra:
            if e in med10 and e in medall and ((med10[e] < 0.05) != (medall[e] < 0.05) or 0.035 <= med10[e] <= 0.07 or 0.035 <= medall[e] <= 0.07):
                near[e][st] = (med10[e], medall[e], len(d[e]))
    if near:
        print("near-threshold epochs inside the window (median p with 10 seeds / with all seeds, * = decision flips):")
        for e in sorted(near):
            cells = []
            for st in ("vrt", "mmd", "ed"):
                if st in near[e]:
                    a, b, n = near[e][st]
                    cells.append(f"{st} {a:.3f}/{b:.3f}{'*' if (a < 0.05) != (b < 0.05) else ' '}[{n}]")
            print(f"  ep {e:4d}  " + "   ".join(cells))
    print()
